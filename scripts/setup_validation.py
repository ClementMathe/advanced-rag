"""
Environment validation script for Advanced RAG system.

This script checks:
- Python version compatibility
- CUDA/GPU availability
- Required dependencies installation
- Directory structure
- Model downloads (optional)

Run this script after initial setup to ensure everything is configured correctly.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class ValidationResult:
    """Container for validation check results."""

    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, message: str = ""):
        """Add a validation check result."""
        self.checks.append((name, passed, message))

    def print_summary(self):
        """Print colored summary of all checks."""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Environment Validation Summary{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")

        passed_count = sum(1 for _, passed, _ in self.checks if passed)
        total_count = len(self.checks)

        for name, passed, message in self.checks:
            status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
            print(f"{status} - {name}")
            if message:
                print(f"       {message}")

        print(f"\n{BLUE}{'='*60}{RESET}")
        if passed_count == total_count:
            print(f"{GREEN}All checks passed ({passed_count}/{total_count}){RESET}")
            print(f"{GREEN}Your environment is ready!{RESET}")
        else:
            print(
                f"{YELLOW}Some checks failed ({passed_count}/{total_count} passed){RESET}"
            )
            print(f"{YELLOW}Please fix the issues above before proceeding.{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")

        return passed_count == total_count


def check_python_version() -> Tuple[bool, str]:
    """
    Verify Python version is 3.10 or higher.

    Returns:
        Tuple of (passed, message)
    """
    version = sys.version_info
    required = (3, 10)

    if version >= required:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor} (requires 3.10+)"


def check_cuda_availability() -> Tuple[bool, str]:
    """
    Check if CUDA is available via torch.

    Returns:
        Tuple of (passed, message)
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version} - {device_name}"
        else:
            return False, "CUDA not available (CPU mode only)"
    except ImportError:
        return False, "torch not installed"


def check_required_packages() -> Tuple[bool, str]:
    """
    Verify all required packages are installed.

    Returns:
        Tuple of (passed, message)
    """
    required_packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "langchain",
        "langgraph",
        "ragas",
        "numpy",
        "pandas",
    ]

    missing = []
    installed = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            installed.append(package)
        except ImportError:
            missing.append(package)

    if not missing:
        return True, f"All {len(installed)} required packages installed"
    else:
        return False, f"Missing packages: {', '.join(missing)}"


def check_directory_structure() -> Tuple[bool, str]:
    """
    Verify project directory structure exists.

    Returns:
        Tuple of (passed, message)
    """
    required_dirs = [
        "src",
        "scripts",
        "data",
        "index",
        "outputs",
        "logs",
        "docs",
        "tests",
    ]

    project_root = Path(__file__).parent.parent
    missing_dirs = []

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            missing_dirs.append(dir_name)

    if not missing_dirs:
        return True, "All directories present"
    else:
        return True, f"Created missing directories: {', '.join(missing_dirs)}"


def check_gpu_memory() -> Tuple[bool, str]:
    """
    Check available GPU memory (if GPU is available).

    Returns:
        Tuple of (passed, message)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return True, "N/A (no GPU)"

        total_memory = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )  # GB

        # Memory requirements with optimizations:
        # - 4GB+: Can run with 4-bit quantization
        # - 6GB+: Comfortable for 4-bit models
        # - 8GB+: Can run fp16 models
        # - 12GB+: Comfortable for fp16 models
        if total_memory < 4:
            return False, f"{total_memory:.1f}GB available (need 4GB+ minimum)"
        elif total_memory < 6:
            return True, f"{total_memory:.1f}GB available (will use 4-bit quantization)"
        elif total_memory < 8:
            return (
                True,
                f"{total_memory:.1f}GB available (optimal for 4-bit quantization)",
            )
        else:
            return True, f"{total_memory:.1f}GB available (can use fp16 models)"
    except Exception as e:
        return False, f"Error checking GPU memory: {str(e)}"


def check_faiss_gpu() -> Tuple[bool, str]:
    """
    Verify FAISS GPU support is available.

    Returns:
        Tuple of (passed, message)
    """
    try:
        import faiss

        # Check if GPU version is installed
        if hasattr(faiss, "StandardGpuResources"):
            try:
                faiss.StandardGpuResources()
                return True, "FAISS GPU support available"
            except Exception:
                return False, "FAISS GPU installed but not functional"
        else:
            # FAISS CPU is acceptable for development/learning
            return (
                True,
                "FAISS CPU version (acceptable for learning, GPU recommended for production)",
            )
    except ImportError:
        return False, "FAISS not installed"


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of key installed packages.

    Returns:
        Dictionary mapping package names to versions
    """
    packages = {
        "torch": None,
        "transformers": None,
        "sentence-transformers": None,
        "langchain": None,
        "langgraph": None,
        "faiss": None,
        "ragas": None,
    }

    for package in packages.keys():
        try:
            module = __import__(package.replace("-", "_"))
            packages[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[package] = "not installed"

    return packages


def print_package_versions():
    """Print versions of all key packages."""
    print(f"\n{BLUE}Installed Package Versions:{RESET}")
    versions = get_package_versions()

    for package, version in versions.items():
        if version == "not installed":
            print(f"  {package}: {RED}{version}{RESET}")
        else:
            print(f"  {package}: {GREEN}{version}{RESET}")


def main():
    """Run all validation checks."""
    print(f"\n{BLUE}Starting environment validation...{RESET}\n")

    result = ValidationResult()

    # Run all checks
    passed, msg = check_python_version()
    result.add("Python Version", passed, msg)

    passed, msg = check_required_packages()
    result.add("Required Packages", passed, msg)

    passed, msg = check_cuda_availability()
    result.add("CUDA Availability", passed, msg)

    passed, msg = check_gpu_memory()
    result.add("GPU Memory", passed, msg)

    passed, msg = check_faiss_gpu()
    result.add("FAISS GPU Support", passed, msg)

    passed, msg = check_directory_structure()
    result.add("Directory Structure", passed, msg)

    # Print results
    all_passed = result.print_summary()

    # Print package versions
    print_package_versions()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
