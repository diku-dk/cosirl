#import os
#import sys
##import site
#
#def setup_sofa_environment(config):
#    """Set up the SOFA Python environment variables and paths."""
#    print(config)
#    # Path to SOFA installation
#    #sofa_root = os.path.expanduser(config["user_settings"]["sofa_path"])
# 
#    # Path to SofaPython3 library
#    #sofa_python3_lib = os.path.join(sofa_root, "plugins/SofaPython3/lib")
#    sofa_python3_lib = os.path.join(os.getcwd(), "SofaPython3/usr/local/lib")
#    
#    # Check for Python bindings directory - could be in python3.X or lib/python3
#    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
#    possible_paths = [
#        os.path.join(sofa_python3_lib, python_version, "site-packages"),
#        os.path.join(sofa_python3_lib, "python3", "site-packages"),
#        sofa_python3_lib,  # Sometimes bindings are directly in lib
#    ]
#    
#    # Find the first valid path
#    sofa_python_path = None
#    for path in possible_paths:
#        if os.path.exists(path):
#            sofa_python_path = path
#            break
#    
#    if not sofa_python_path:
#        raise RuntimeError(f"Could not find SofaPython3 bindings in {sofa_python3_lib}")
#    
#    # Add to Python path if it's not already there
#    if sofa_python_path not in sys.path:
#        sys.path.insert(0, sofa_python_path)
#    
#    # Also add SOFA plugins to LD_LIBRARY_PATH if not running
#    lib_path = os.path.join(sofa_root, "lib")
#    os_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
#    if lib_path not in os_lib_path:
#        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os_lib_path}"
#    
#    # Verify the setup worked
#    try:
#        import Sofa
#        import SofaRuntime
#        print("SofaPython3 environment successfully configured!")
#    except ImportError as e:
#        print(f"Failed to import SOFA modules: {e}")
#        print(f"PYTHONPATH: {sys.path}")
#        raise

import os
import sys

def setup_sofa_environment(config):
    """Set up the SOFA Python environment variables and paths."""
    print(config)
    
    # Main paths
    sofa_root = os.path.expanduser(config["user_settings"]["sofa_path"])
    repo_root = os.getcwd()
    sofapython3_root = os.path.join(repo_root, "SofaPython3/usr/local")
    
    # Python bindings directory
    sofapython3_lib = os.path.join(sofapython3_root, "lib")
    sofapython3_python_path = os.path.join(sofapython3_lib, "python3", "site-packages")
    
    # If site-packages doesn't exist, check if the .so files are directly in lib
    if not os.path.exists(sofapython3_python_path):
        # Look for .so files directly in lib
        if any(f.endswith(".so") for f in os.listdir(sofapython3_lib)):
            sofapython3_python_path = sofapython3_lib
    
    # Add to Python path if it's not already there
    if sofapython3_python_path not in sys.path:
        sys.path.insert(0, sofapython3_python_path)
    
    # Set environment variables
    os.environ["SOFA_ROOT"] = sofa_root
    
    # Add library paths to LD_LIBRARY_PATH
    lib_paths = [
        os.path.join(sofa_root, "lib"),
        sofapython3_lib
    ]
    
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            os_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_path not in os_lib_path:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os_lib_path}"
    
    # Debugging info
    print(f"SOFA_ROOT set to: {os.environ.get('SOFA_ROOT')}")
    print(f"Python paths: {sys.path[:3]}")  # Just show first few paths
    
    # Verify the setup worked
    try:
        import Sofa
        import SofaRuntime
        print("SofaPython3 environment successfully configured!")
    except ImportError as e:
        print(f"Failed to import SOFA modules: {e}")
        print(f"PYTHONPATH: {sys.path}")
        
        # Check what files are in the Python path
        if os.path.exists(sofapython3_python_path):
            print(f"Files in {sofapython3_python_path}:")
            print(os.listdir(sofapython3_python_path))
        
        # Try to directly import the .so files
        if os.path.exists(sofapython3_lib):
            print(f"Trying to import from .so files in {sofapython3_lib}")
            sys.path.insert(0, sofapython3_lib)
            try:
                import SofaRuntime
                print("Successfully imported SofaRuntime from .so files")
            except ImportError as e2:
                print(f"Still failed: {e2}")
        
        raise
