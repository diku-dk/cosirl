import os
import sys
#import site

def setup_sofa_environment(config):
    """Set up the SOFA Python environment variables and paths."""
    print(config)
    # Path to SOFA installation
    sofa_root = os.path.expanduser(config["user_settings"]["sofa_path"])
 
    # Path to SofaPython3 library
    sofa_python3_lib = os.path.join(sofa_root, "plugins/SofaPython3/lib")
    
    # Check for Python bindings directory - could be in python3.X or lib/python3
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    possible_paths = [
        os.path.join(sofa_python3_lib, python_version, "site-packages"),
        os.path.join(sofa_python3_lib, "python3", "site-packages"),
        sofa_python3_lib,  # Sometimes bindings are directly in lib
    ]
    
    # Find the first valid path
    sofa_python_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sofa_python_path = path
            break
    
    if not sofa_python_path:
        raise RuntimeError(f"Could not find SofaPython3 bindings in {sofa_python3_lib}")
    
    # Add to Python path if it's not already there
    if sofa_python_path not in sys.path:
        sys.path.insert(0, sofa_python_path)
    
    # Also add SOFA plugins to LD_LIBRARY_PATH if not running
    lib_path = os.path.join(sofa_root, "lib")
    os_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in os_lib_path:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os_lib_path}"
    
    # Verify the setup worked
    try:
        import Sofa
        import SofaRuntime
        print("SofaPython3 environment successfully configured!")
    except ImportError as e:
        print(f"Failed to import SOFA modules: {e}")
        print(f"PYTHONPATH: {sys.path}")
        raise

