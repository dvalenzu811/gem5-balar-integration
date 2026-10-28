import os
import modules


class Benchmark(modules.Benchmark):
    name = "vecadd"

    def get_parameters(self, options):
        # No benchmark-specific sweep parameters for now.
        return {}

    def get_command(self, parameters: dict, conf_dict: dict):
        # Full-system path: executable is inside the benchmark disk.
        return os.path.join(self.executable_dir(), "vecadd-ubuntu")

    def parse_roi(self, rdict: dict, conf_dict: dict):
        rdict = super().parse_roi(rdict, conf_dict)
        rdict["kernel"] = self.name
        return rdict

    def get_cuda_executable(self, parameters: dict, conf_dict: dict):
        # Host-visible ELF used by Gem5Balar/GPGPU-Sim.
        return os.path.join(self.binary_dir, "vecadd-ubuntu")

    def get_host_workdir(self, parameters: dict, conf_dict: dict):
        # GPGPU-Sim loads gpgpusim.config relative to its host CWD.
        #
        # By default vecadd's binary_dir is:
        #
        #   .../balar/tests/vanadis_llvm_rv64
        #
        # therefore the parent directory is:
        #
        #   .../balar/tests
        #
        # where gpgpusim.config is staged.
        return os.environ.get(
            "BALAR_GPGPUSIM_WORKDIR",
            os.path.dirname(self.binary_dir),
        )

    def get_host_setup(self, parameters: dict, conf_dict: dict):
        # Host setup is deliberately kept outside the benchmark.
        #
        # setup_env.sh is responsible for machine-specific paths,
        # modules, CUDA, GPGPU-Sim libraries and parser Python.
        setup = os.environ.get("BALAR_HOST_SETUP_SCRIPT", "")

        if not setup:
            raise RuntimeError(
                "BALAR_HOST_SETUP_SCRIPT is not defined. "
                "Source scripts/setup_env.sh before generating vecadd."
            )

        setup = os.path.abspath(setup)

        return f"""test -f "{setup}" || {{
    echo "ERROR: missing Gem5Balar host setup: {setup}"
    exit 1
}}

source "{setup}"
"""
