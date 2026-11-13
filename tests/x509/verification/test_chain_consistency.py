# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

"""Consistency tests for X.509 path construction and extension evaluation.

These tests use locally generated, deterministic certificate fixtures
(fixed serial numbers, names, validity windows and key usages) to assert
that verification verdicts are independent of the *ordering* of the
untrusted intermediate certificate list, and that every relevant extension
and validity check jointly contributes to the result:

* validity window (inclusive clock boundaries);
* subjectAltName matching and name constraints (permitted/excluded);
* basicConstraints (cA and path length);
* keyUsage and extendedKeyUsage;
* unrecognized critical extensions;
* duplicate certificates, loops, empty/untrusted chains;
* cross-signing and fatal-vs-branch error handling;
* concurrent verification stability.

The tests are written with :mod:`unittest` so they can be driven both by
``python -m unittest`` and by pytest.
"""

from __future__ import annotations

import datetime
import threading
import unittest
from functools import cache
from itertools import permutations

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.general_name import DNSName
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from cryptography.x509.verification import (
    PolicyBuilder,
    Store,
    VerificationError,
)

VALIDATION_TIME = datetime.datetime(2025, 6, 1, 12, 0, 0)
NOT_BEFORE = datetime.datetime(2024, 1, 1, 0, 0, 0)
NOT_AFTER = datetime.datetime(2030, 1, 1, 0, 0, 0)

CA_KEY_USAGE = x509.KeyUsage(
    digital_signature=False,
    content_commitment=False,
    key_encipherment=False,
    data_encipherment=False,
    key_agreement=False,
    key_cert_sign=True,
    crl_sign=True,
    encipher_only=False,
    decipher_only=False,
)


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _ca_cert(
    key,
    subject: x509.Name,
    issuer_key,
    issuer_name: x509.Name,
    serial: int,
    *,
    extra_extensions=(),
    validity=(NOT_BEFORE, NOT_AFTER),
    path_length=None,
    ca: bool = True,
):
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(serial)
        .not_valid_before(validity[0])
        .not_valid_after(validity[1])
        .add_extension(
            x509.BasicConstraints(ca=ca, path_length=path_length),
            critical=True,
        )
        .add_extension(CA_KEY_USAGE, critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
    )
    for extension, critical in extra_extensions:
        builder = builder.add_extension(extension, critical=critical)
    return builder.sign(issuer_key, hashes.SHA256())


def _leaf_cert(
    key,
    cn: str,
    issuer_key,
    issuer_name: x509.Name,
    serial: int,
    sans,
    *,
    extra_extensions=(),
    validity=(NOT_BEFORE, NOT_AFTER),
    add_aki: bool = True,
):
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(serial)
        .not_valid_before(validity[0])
        .not_valid_after(validity[1])
        .add_extension(x509.SubjectAlternativeName(list(sans)), critical=False)
    )
    if add_aki:
        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(
                issuer_key.public_key()
            ),
            critical=False,
        )
    for extension, critical in extra_extensions:
        builder = builder.add_extension(extension, critical=critical)
    return builder.sign(issuer_key, hashes.SHA256())


def _server_verifier(
    store: Store, *, time=VALIDATION_TIME, subject="example.com"
):
    return (
        PolicyBuilder()
        .store(store)
        .time(time)
        .build_server_verifier(DNSName(subject))
    )


def _client_verifier(store: Store, *, time=VALIDATION_TIME):
    return PolicyBuilder().store(store).time(time).build_client_verifier()


class _Fixtures:
    """Deterministic certificate hierarchy."""

    def __init__(self) -> None:
        # Simple root -> intermediate -> leaf chain.
        self.root_key = ec.generate_private_key(ec.SECP256R1())
        self.int_key = ec.generate_private_key(ec.SECP256R1())
        self.leaf_key = ec.generate_private_key(ec.SECP256R1())
        self.root = _ca_cert(
            self.root_key, _name("Root CA"), self.root_key, _name("Root CA"), 1
        )
        self.intermediate = _ca_cert(
            self.int_key,
            _name("Intermediate CA"),
            self.root_key,
            _name("Root CA"),
            2,
        )
        self.leaf = _leaf_cert(
            self.leaf_key,
            "example.com",
            self.int_key,
            _name("Intermediate CA"),
            3,
            [x509.DNSName("example.com")],
        )

        # A second, distinct root sharing no key material.
        self.other_root_key = ec.generate_private_key(ec.SECP256R1())
        self.other_root = _ca_cert(
            self.other_root_key,
            _name("Other Root CA"),
            self.other_root_key,
            _name("Other Root CA"),
            4,
        )


@cache
def _fixtures() -> _Fixtures:
    return _Fixtures()


class TestOrderIndependentChains(unittest.TestCase):
    """The verdict must not depend on intermediate list ordering."""

    def _assert_verdict_stable(self, verifier, leaf, intermediates):
        verdicts = set()
        for ordering in permutations(intermediates):
            try:
                verifier.verify(leaf, list(ordering))
            except VerificationError as error:
                verdicts.add(("reject", str(error)))
            else:
                verdicts.add(("accept", ""))
        self.assertEqual(
            len(verdicts),
            1,
            f"verdict depends on intermediate ordering: {verdicts!r}",
        )
        return next(iter(verdicts))

    def test_chain_accepts_in_every_intermediate_order(self):
        verifier = _server_verifier(Store([_fixtures().root]))
        verdict = self._assert_verdict_stable(
            verifier, _fixtures().leaf, [_fixtures().intermediate]
        )
        self.assertEqual(verdict[0], "accept")

    def test_duplicate_intermediate_entries_accepted(self):
        # Repeating the same certificate in the untrusted list must not
        # change the verdict nor cause pathological behavior.
        verifier = _server_verifier(Store([_fixtures().root]))
        chain = verifier.verify(
            _fixtures().leaf,
            [_fixtures().intermediate, _fixtures().intermediate],
        )
        self.assertEqual(
            chain,
            [_fixtures().leaf, _fixtures().intermediate, _fixtures().root],
        )

    def test_leaf_listed_among_intermediates_is_harmless(self):
        verifier = _server_verifier(Store([_fixtures().root]))
        chain = verifier.verify(
            _fixtures().leaf, [_fixtures().leaf, _fixtures().intermediate]
        )
        self.assertEqual(chain[0], _fixtures().leaf)

    def test_unrelated_intermediate_does_not_relax_validation(self):
        # A self-signed CA that is not in the trust store must never be
        # usable as an anchor, regardless of where it appears.
        loop_key = ec.generate_private_key(ec.SECP256R1())
        loop_ca = _ca_cert(
            loop_key, _name("Loop CA"), loop_key, _name("Loop CA"), 10
        )
        verifier = _server_verifier(Store([_fixtures().other_root]))
        for ordering in (
            [loop_ca, _fixtures().intermediate],
            [_fixtures().intermediate, loop_ca],
            [loop_ca, loop_ca],
        ):
            with self.assertRaises(VerificationError):
                verifier.verify(_fixtures().leaf, ordering)

    def test_cross_signed_intermediate_order_independence(self):
        # The intermediate's key and subject are cross-signed by two roots.
        # Both roots are trusted, so verification must succeed no matter
        # which cross-signed variant appears first.
        alt_root_key = ec.generate_private_key(ec.SECP256R1())
        alt_root = _ca_cert(
            alt_root_key,
            _name("Root CA"),
            alt_root_key,
            _name("Root CA"),
            20,
        )
        cross_signed = _ca_cert(
            _fixtures().int_key,
            _name("Intermediate CA"),
            alt_root_key,
            _name("Root CA"),
            21,
        )
        verifier = _server_verifier(Store([_fixtures().root, alt_root]))
        verdict = self._assert_verdict_stable(
            verifier,
            _fixtures().leaf,
            [_fixtures().intermediate, cross_signed],
        )
        self.assertEqual(verdict[0], "accept")

    def test_fatal_branch_does_not_abort_alternate_path(self):
        # Regression test: one cross-signed variant carries a critical
        # nameConstraints extension with an unsupported directoryName
        # constraint. That disqualifies only that candidate; the clean
        # variant must still produce a successful chain, and the verdict
        # must be identical regardless of input ordering.
        bad_root_key = ec.generate_private_key(ec.SECP256R1())
        bad_root = _ca_cert(
            bad_root_key,
            _name("Bad Root CA"),
            bad_root_key,
            _name("Bad Root CA"),
            30,
        )
        directory_name = x509.DirectoryName(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PyCA")])
        )
        unsupported_nc = x509.NameConstraints(
            permitted_subtrees=[directory_name], excluded_subtrees=None
        )
        bad_variant = _ca_cert(
            _fixtures().int_key,
            _name("Intermediate CA"),
            bad_root_key,
            _name("Bad Root CA"),
            31,
            extra_extensions=[(unsupported_nc, True)],
        )
        verifier = _server_verifier(Store([_fixtures().root, bad_root]))
        verdict = self._assert_verdict_stable(
            verifier, _fixtures().leaf, [bad_variant, _fixtures().intermediate]
        )
        self.assertEqual(verdict[0], "accept")

    def test_unsupported_constraint_still_rejects_without_alternate(self):
        # If every candidate carries the unsupported critical constraint,
        # validation must still fail (the branch error is not swallowed).
        directory_name = x509.DirectoryName(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PyCA")])
        )
        unsupported_nc = x509.NameConstraints(
            permitted_subtrees=[directory_name], excluded_subtrees=None
        )
        constrained_root = _ca_cert(
            _fixtures().root_key,
            _name("Constrained Root CA"),
            _fixtures().root_key,
            _name("Constrained Root CA"),
            40,
            extra_extensions=[(unsupported_nc, True)],
        )
        # A leaf issued directly by the constrained root (no intermediate).
        leaf = _leaf_cert(
            _fixtures().leaf_key,
            "example.com",
            _fixtures().root_key,
            _name("Constrained Root CA"),
            41,
            [x509.DNSName("example.com")],
        )
        verifier = _server_verifier(Store([constrained_root]))
        with self.assertRaisesRegex(
            VerificationError, "unsupported directoryName constraint"
        ):
            verifier.verify(leaf, [])


class _NameConstraintMaterial:
    def __init__(self) -> None:
        self.ca_key = ec.generate_private_key(ec.SECP256R1())
        self.leaf_key = ec.generate_private_key(ec.SECP256R1())
        self.ca_name = _name("Constrained CA")


@cache
def _name_constraint_material() -> _NameConstraintMaterial:
    return _NameConstraintMaterial()


class TestNameConstraints(unittest.TestCase):
    def _ca_with_constraints(
        self, name_constraints, *, critical=True, serial=50
    ):
        material = _name_constraint_material()
        return _ca_cert(
            material.ca_key,
            material.ca_name,
            material.ca_key,
            material.ca_name,
            serial,
            extra_extensions=[(name_constraints, critical)],
        )

    def _leaf(self, sans, serial=51):
        material = _name_constraint_material()
        return _leaf_cert(
            material.leaf_key,
            "leaf",
            material.ca_key,
            material.ca_name,
            serial,
            sans,
        )

    def test_permitted_subtree_rejects_outside_name(self):
        ca = self._ca_with_constraints(
            x509.NameConstraints(
                permitted_subtrees=[x509.DNSName("allowed.test")],
                excluded_subtrees=None,
            )
        )
        leaf = self._leaf([x509.DNSName("forbidden.test")])
        with self.assertRaisesRegex(
            VerificationError, "no permitted name constraints matched SAN"
        ):
            _client_verifier(Store([ca])).verify(leaf, [])

    def test_permitted_subtree_accepts_name_within_subtree(self):
        ca = self._ca_with_constraints(
            x509.NameConstraints(
                permitted_subtrees=[x509.DNSName("allowed.test")],
                excluded_subtrees=None,
            ),
            serial=52,
        )
        leaf = self._leaf(
            [x509.DNSName("allowed.test"), x509.DNSName("host.allowed.test")],
            serial=53,
        )
        _client_verifier(Store([ca])).verify(leaf, [])

    def test_excluded_subtree_rejects_matching_name(self):
        ca = self._ca_with_constraints(
            x509.NameConstraints(
                permitted_subtrees=None,
                excluded_subtrees=[x509.DNSName("banned.allowed.test")],
            ),
            serial=54,
        )
        leaf = self._leaf([x509.DNSName("banned.allowed.test")], serial=55)
        with self.assertRaisesRegex(
            VerificationError, "excluded name constraint matched SAN"
        ):
            _client_verifier(Store([ca])).verify(leaf, [])

    def test_constraint_and_san_must_both_hold(self):
        # A name that satisfies the SAN/subject requirement but violates
        # the name constraint must never be accepted (joint evaluation).
        ca = self._ca_with_constraints(
            x509.NameConstraints(
                permitted_subtrees=[x509.DNSName("allowed.test")],
                excluded_subtrees=None,
            ),
            serial=56,
        )
        leaf = self._leaf([x509.DNSName("example.com")], serial=57)
        with self.assertRaises(VerificationError):
            _server_verifier(Store([ca])).verify(leaf, [])
        # A leaf inside the constrained subtree but verified against a
        # different subject fails the SAN match instead.
        allowed_leaf = self._leaf(
            [x509.DNSName("host.allowed.test")], serial=58
        )
        with self.assertRaisesRegex(
            VerificationError, "no matching subjectAltName"
        ):
            _server_verifier(Store([ca]), subject="example.com").verify(
                allowed_leaf, []
            )


class _ExtensionUsageMaterial:
    def __init__(self) -> None:
        self.ca_key = ec.generate_private_key(ec.SECP256R1())
        self.leaf_key = ec.generate_private_key(ec.SECP256R1())
        self.ca_name = _name("Policy CA")
        self.ca = _ca_cert(
            self.ca_key, self.ca_name, self.ca_key, self.ca_name, 60
        )


@cache
def _extension_usage_material() -> _ExtensionUsageMaterial:
    return _ExtensionUsageMaterial()


class TestExtensionAndUsage(unittest.TestCase):
    def _leaf(self, extra_extensions, serial=61, **kwargs):
        material = _extension_usage_material()
        return _leaf_cert(
            material.leaf_key,
            "example.com",
            material.ca_key,
            material.ca_name,
            serial,
            [x509.DNSName("example.com")],
            extra_extensions=extra_extensions,
            **kwargs,
        )

    def test_leaf_with_key_cert_sign_rejected(self):
        bad_ku = x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=True,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        )
        leaf = self._leaf([(bad_ku, True)])
        with self.assertRaisesRegex(VerificationError, "keyCertSign"):
            _server_verifier(Store([_extension_usage_material().ca])).verify(
                leaf, []
            )

    def test_ca_without_key_cert_sign_rejected(self):
        weak_ca_key = ec.generate_private_key(ec.SECP256R1())
        weak_ca = (
            x509.CertificateBuilder()
            .subject_name(_name("Weak CA"))
            .issuer_name(_name("Weak CA"))
            .public_key(weak_ca_key.public_key())
            .serial_number(70)
            .not_valid_before(NOT_BEFORE)
            .not_valid_after(NOT_AFTER)
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None), critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ),
                critical=True,
            )
            .sign(weak_ca_key, hashes.SHA256())
        )
        leaf = _leaf_cert(
            _extension_usage_material().leaf_key,
            "example.com",
            weak_ca_key,
            _name("Weak CA"),
            71,
            [x509.DNSName("example.com")],
        )
        with self.assertRaisesRegex(
            VerificationError, "keyUsage.keyCertSign must be asserted"
        ):
            _server_verifier(Store([weak_ca])).verify(leaf, [])

    def test_leaf_eku_mismatch_rejected(self):
        bad_eku = x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION])
        leaf = self._leaf([(bad_eku, False)])
        with self.assertRaisesRegex(
            VerificationError, "required EKU not found"
        ):
            _server_verifier(Store([_extension_usage_material().ca])).verify(
                leaf, []
            )
        with self.assertRaisesRegex(
            VerificationError, "required EKU not found"
        ):
            _client_verifier(Store([_extension_usage_material().ca])).verify(
                leaf, []
            )

    def test_leaf_eku_match_accepted(self):
        good_eku = x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH])
        leaf = self._leaf([(good_eku, False)], serial=72)
        _server_verifier(Store([_extension_usage_material().ca])).verify(
            leaf, []
        )

    def test_ca_basic_constraints_ca_false_rejected(self):
        not_ca_key = ec.generate_private_key(ec.SECP256R1())
        not_ca = _ca_cert(
            not_ca_key,
            _name("Not Actually A CA"),
            not_ca_key,
            _name("Not Actually A CA"),
            80,
            ca=False,
        )
        leaf = _leaf_cert(
            _extension_usage_material().leaf_key,
            "example.com",
            not_ca_key,
            _name("Not Actually A CA"),
            81,
            [x509.DNSName("example.com")],
        )
        with self.assertRaisesRegex(
            VerificationError, "basicConstraints.cA must be asserted"
        ):
            _server_verifier(Store([not_ca])).verify(leaf, [])

    def test_path_length_constraint_enforced(self):
        # The intermediate CA asserts pathLength=0, so a second
        # intermediate below it exceeds the permitted path.
        root_key = ec.generate_private_key(ec.SECP256R1())
        int1_key = ec.generate_private_key(ec.SECP256R1())
        int2_key = ec.generate_private_key(ec.SECP256R1())
        root = _ca_cert(
            root_key, _name("PL Root"), root_key, _name("PL Root"), 90
        )
        int1 = _ca_cert(
            int1_key,
            _name("PL Int 1"),
            root_key,
            _name("PL Root"),
            91,
            path_length=0,
        )
        int2 = _ca_cert(
            int2_key,
            _name("PL Int 2"),
            int1_key,
            _name("PL Int 1"),
            92,
        )
        leaf = _leaf_cert(
            _extension_usage_material().leaf_key,
            "example.com",
            int2_key,
            _name("PL Int 2"),
            93,
            [x509.DNSName("example.com")],
        )
        with self.assertRaisesRegex(
            VerificationError, "path length constraint violated"
        ):
            _server_verifier(Store([root])).verify(leaf, [int1, int2])
        # The same verdict must hold for the reversed ordering.
        with self.assertRaisesRegex(
            VerificationError, "path length constraint violated"
        ):
            _server_verifier(Store([root])).verify(leaf, [int2, int1])

    def test_unrecognized_critical_extension_rejected(self):
        unknown = x509.UnrecognizedExtension(
            x509.ObjectIdentifier("1.2.3.4.5.6.7.8.9"), b"\x04\x03abc"
        )
        leaf = self._leaf([(unknown, True)], serial=93)
        with self.assertRaisesRegex(
            VerificationError, "unaccounted-for critical extensions"
        ):
            _server_verifier(Store([_extension_usage_material().ca])).verify(
                leaf, []
            )

    def test_non_critical_unrecognized_extension_accepted(self):
        unknown = x509.UnrecognizedExtension(
            x509.ObjectIdentifier("1.2.3.4.5.6.7.8.10"), b"\x04\x03abc"
        )
        leaf = self._leaf([(unknown, False)], serial=94)
        _server_verifier(Store([_extension_usage_material().ca])).verify(
            leaf, []
        )

    def test_san_subject_mismatch_rejected(self):
        # A SAN that does not match the verified subject.
        leaf = _leaf_cert(
            _extension_usage_material().leaf_key,
            "example.com",
            _extension_usage_material().ca_key,
            _extension_usage_material().ca_name,
            95,
            [x509.DNSName("different.example")],
        )
        with self.assertRaisesRegex(
            VerificationError, "no matching subjectAltName"
        ):
            _server_verifier(
                Store([_extension_usage_material().ca]), subject="example.com"
            ).verify(leaf, [])


class _ValidityMaterial:
    def __init__(self) -> None:
        self.ca_key = ec.generate_private_key(ec.SECP256R1())
        self.leaf_key = ec.generate_private_key(ec.SECP256R1())
        self.not_before = datetime.datetime(2025, 1, 1, 0, 0, 0)
        self.not_after = datetime.datetime(2025, 12, 31, 23, 59, 59)
        self.ca = _ca_cert(
            self.ca_key,
            _name("Validity CA"),
            self.ca_key,
            _name("Validity CA"),
            100,
            validity=(self.not_before, self.not_after),
        )
        self.leaf = _leaf_cert(
            self.leaf_key,
            "example.com",
            self.ca_key,
            _name("Validity CA"),
            101,
            [x509.DNSName("example.com")],
            validity=(self.not_before, self.not_after),
        )


@cache
def _validity_material() -> _ValidityMaterial:
    return _ValidityMaterial()


class TestValidityBoundaries(unittest.TestCase):
    def test_not_before_is_inclusive(self):
        material = _validity_material()
        _server_verifier(
            Store([material.ca]), time=material.not_before
        ).verify(material.leaf, [])

    def test_not_after_is_inclusive(self):
        material = _validity_material()
        _server_verifier(Store([material.ca]), time=material.not_after).verify(
            material.leaf, []
        )

    def test_one_second_before_not_before_rejected(self):
        material = _validity_material()
        with self.assertRaisesRegex(
            VerificationError, "not valid at validation time"
        ):
            _server_verifier(
                Store([material.ca]),
                time=material.not_before - datetime.timedelta(seconds=1),
            ).verify(material.leaf, [])

    def test_one_second_after_not_after_rejected(self):
        material = _validity_material()
        with self.assertRaisesRegex(
            VerificationError, "not valid at validation time"
        ):
            _server_verifier(
                Store([material.ca]),
                time=material.not_after + datetime.timedelta(seconds=1),
            ).verify(material.leaf, [])

    def test_expired_anchor_cannot_sign_valid_leaf(self):
        # An expired issuer is checked when it validates the child, so
        # the chain fails even though the leaf itself is in date.
        material = _validity_material()
        expired_root_key = ec.generate_private_key(ec.SECP256R1())
        expired_root = _ca_cert(
            expired_root_key,
            _name("Expired Root"),
            expired_root_key,
            _name("Expired Root"),
            110,
            validity=(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2022, 1, 1),
            ),
        )
        leaf = _leaf_cert(
            material.leaf_key,
            "example.com",
            expired_root_key,
            _name("Expired Root"),
            111,
            [x509.DNSName("example.com")],
        )
        with self.assertRaisesRegex(
            VerificationError, "not valid at validation time"
        ):
            _server_verifier(
                Store([expired_root]),
                time=datetime.datetime(2025, 1, 1),
            ).verify(leaf, [])


class TestTrustAndEmptyChains(unittest.TestCase):
    def test_empty_intermediates_untrusted_leaf_rejected(self):
        # The leaf is not itself a trust anchor and nothing is provided
        # to build a path to one.
        verifier = _server_verifier(Store([_fixtures().other_root]))
        with self.assertRaises(VerificationError):
            verifier.verify(_fixtures().leaf, [])

    def test_untrusted_root_is_not_anchor(self):
        verifier = _server_verifier(Store([_fixtures().other_root]))
        with self.assertRaises(VerificationError):
            verifier.verify(_fixtures().leaf, [_fixtures().intermediate])

    def test_trusted_anchor_directly_issues_leaf(self):
        # No intermediates: the leaf is signed directly by the anchor.
        leaf = _leaf_cert(
            _fixtures().leaf_key,
            "example.com",
            _fixtures().root_key,
            _name("Root CA"),
            120,
            [x509.DNSName("example.com")],
        )
        chain = _server_verifier(Store([_fixtures().root])).verify(leaf, [])
        self.assertEqual(chain, [leaf, _fixtures().root])


class TestConcurrentVerification(unittest.TestCase):
    def test_concurrent_verification_is_stable(self):
        fx = _Fixtures()
        verifier = _server_verifier(Store([fx.root]))
        results = []
        errors = []

        def worker():
            try:
                for _ in range(50):
                    chain = verifier.verify(fx.leaf, [fx.intermediate])
                    results.append(len(chain))
            except (
                VerificationError
            ) as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertTrue(results)
        self.assertEqual(set(results), {3})

    def test_concurrent_cross_signed_ordering_is_stable(self):
        fx = _Fixtures()
        alt_root_key = ec.generate_private_key(ec.SECP256R1())
        alt_root = _ca_cert(
            alt_root_key,
            _name("Root CA"),
            alt_root_key,
            _name("Root CA"),
            130,
        )
        cross_signed = _ca_cert(
            fx.int_key,
            _name("Intermediate CA"),
            alt_root_key,
            _name("Root CA"),
            131,
        )
        verifier = _server_verifier(Store([fx.root, alt_root]))
        orderings = (
            [fx.intermediate, cross_signed],
            [cross_signed, fx.intermediate],
        )
        outcomes = []
        lock = threading.Lock()

        def worker(ordering):
            try:
                verifier.verify(fx.leaf, ordering)
                outcome = "accept"
            except VerificationError as error:
                outcome = f"reject: {error}"
            with lock:
                outcomes.append(outcome)

        threads = []
        for _ in range(10):
            threads.append(
                threading.Thread(target=worker, args=(orderings[_ % 2],))
            )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(set(outcomes), {"accept"})


class TestErrorStability(unittest.TestCase):
    def test_error_message_is_stable_across_orderings(self):
        # When every ordering fails (untrusted anchor), the error class is
        # always VerificationError regardless of ordering.
        verifier = _server_verifier(Store([_fixtures().other_root]))
        for ordering in (
            [_fixtures().intermediate],
            [_fixtures().intermediate, _fixtures().intermediate],
        ):
            with self.assertRaises(VerificationError):
                verifier.verify(_fixtures().leaf, ordering)

    def test_error_never_contains_private_key_material(self):
        # Serialize the leaf private key and ensure no error rendering
        # ever includes any portion of it.
        pem = _fixtures().leaf_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        der = _fixtures().leaf_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        verifier = _server_verifier(Store([_fixtures().other_root]))
        try:
            verifier.verify(_fixtures().leaf, [_fixtures().intermediate])
        except VerificationError as error:
            message = str(error)
        else:  # pragma: no cover - this chain is expected to fail
            self.fail("verification unexpectedly succeeded")

        # Neither the PEM framing nor any of its base64 body may leak.
        self.assertNotIn("PRIVATE KEY", message)
        pem_body = b"".join(pem.splitlines()[1:-1])
        self.assertNotIn(pem_body, message.encode())
        # No contiguous slice of the DER-encoded private key may appear in
        # the error message.
        for size in (8, 16, 32):
            for index in range(0, len(der) - size, size):
                self.assertNotIn(der[index : index + size], message.encode())


if __name__ == "__main__":
    unittest.main()
