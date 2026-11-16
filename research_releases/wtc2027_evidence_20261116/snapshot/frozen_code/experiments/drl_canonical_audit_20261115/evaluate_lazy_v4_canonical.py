"""Independent audited evaluator for lazy-certificate v4 checkpoints."""

import audit_common


def install_lazy_patches() -> None:
    from clock_patch import install as install_clock
    from timing_candidate import install as install_timing
    from completion_guard import install as install_completion
    from admission_lazy_v4 import install as install_admission
    from depot_noop import install as install_depot_noop
    from lazy_decoder_v4 import install as install_decoder

    install_clock()
    install_timing()
    install_completion()
    install_admission()
    install_depot_noop()
    install_decoder()


audit_common.install_deployment_patches = install_lazy_patches

import evaluate_canonical as base  # noqa: E402


base.install_deployment_patches = install_lazy_patches
base.ADMISSION_MODULE = "admission_lazy_v4"
base.EXPECTED_VERSIONS = {
    "canonical_training_version": 2,
    "cost_semantics_version": 4,
    "dispatch_clock_version": 1,
    "completion_guard_version": 1,
    "hard4_admission_version": 4,
    "lazy_certificate_decoder_version": 4,
    "depot_noop_version": 1,
}


if __name__ == "__main__":
    base.main()
