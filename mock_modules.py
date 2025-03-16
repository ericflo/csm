#\!/usr/bin/env python
"""
Mock modules for missing dependencies - use before any other imports.
"""

import sys
import importlib

# Create dummy ModuleSpec for use in DummyModule
class DummyModuleSpec:
    """Simple ModuleSpec implementation for dummy modules."""
    def __init__(self, name):
        self.name = name
        self.loader = None
        self.origin = f"<dummy-module-spec-{name}>"
        self.submodule_search_locations = []
        self.parent = name.rsplit('.', 1)[0] if '.' in name else None
        self.has_location = False
        self.cached = False
        
    def __repr__(self):
        return f"<DummyModuleSpec '{self.name}'>"

# Create robust dummy module
class DummyModule:
    """Robust dummy module implementation that can be used as a drop-in replacement."""
    def __init__(self, name):
        self.name = name
        self.__all__ = []
        self.__path__ = []
        self.__file__ = f"<dummy-module-{name}>"
        # Create a proper spec object rather than None to fix importlib issues
        self.__spec__ = DummyModuleSpec(name)
        self.__package__ = name
        self.__loader__ = None
        
    def __getattr__(self, name):
        # Create submodules on demand
        submodule = DummyModule(f"{self.name}.{name}")
        # Store for consistency
        setattr(self, name, submodule)
        return submodule
        
    def __call__(self, *args, **kwargs):
        # Act as a callable that returns None
        return None
        
    def __iter__(self):
        # Empty iterator 
        return iter([])
    
    def __len__(self):
        # Empty module has zero length
        return 0
        
    def __bool__(self):
        # Boolean context evaluates to False
        return False
        
    def __dir__(self):
        # Provide a minimal set of attributes for dir()
        return ['__all__', '__file__', '__path__', '__spec__', '__package__', '__loader__']

# Install dummy modules
def install_dummy_modules(module_list):
    """Install dummy modules for missing dependencies.
    
    Args:
        module_list: List of module names to replace with dummies
    """
    for module_name in module_list:
        if module_name not in sys.modules:
            print(f"Installing dummy module for missing dependency: {module_name}")
            sys.modules[module_name] = DummyModule(module_name)
            
            # Add common submodules for known dependencies
            if module_name == 'triton':
                sys.modules['triton.language'] = DummyModule('triton.language')
                sys.modules['triton.ops'] = DummyModule('triton.ops')
                # triton.language is often imported as tl
                sys.modules['tl'] = sys.modules['triton.language']
            elif module_name == 'bitsandbytes':
                sys.modules['bitsandbytes.nn'] = DummyModule('bitsandbytes.nn')
                sys.modules['bitsandbytes.functional'] = DummyModule('bitsandbytes.functional')

# Standard list of modules to mock for ML projects
STANDARD_MOCK_MODULES = [
    'triton',
    'bitsandbytes',
    'flash_attn',
    'xformers'
]

# Handle the transformers package specifically 
def patch_transformers():
    """Patch transformers package to handle missing dependencies."""
    try:
        # More comprehensive patching to handle transformers
        
        # First ensure our mocked modules are perfect
        for module_name in STANDARD_MOCK_MODULES:
            if module_name in sys.modules:
                # Make sure the module has a proper spec to avoid importlib issues
                spec_module = sys.modules[module_name]
                if not hasattr(spec_module, '__spec__') or spec_module.__spec__ is None:
                    spec_module.__spec__ = DummyModuleSpec(module_name)
                
        # Method 1: Patch transformers.utils.import_utils directly
        try:
            import transformers.utils.import_utils as import_utils
            
            # Save the original _is_package_available function
            original_is_pkg_available = import_utils._is_package_available
            
            # Create a patched version that handles our dummy modules
            def patched_is_pkg_available(pkg_name):
                if pkg_name in STANDARD_MOCK_MODULES:
                    print(f"Transformers checking for {pkg_name} - reporting as unavailable")
                    return False
                try:
                    return original_is_pkg_available(pkg_name)
                except Exception:
                    # If the original function fails, just report the package as unavailable
                    return False
            
            # Apply the patch
            import_utils._is_package_available = patched_is_pkg_available
            print("Patched transformers.utils.import_utils._is_package_available")
            
            # Set availability flags directly to avoid checks
            import_utils._torch_available = True
            import_utils._torch_version = "2.0.0"  # Set a reasonable version
            import_utils._bitsandbytes_available = False
            import_utils._triton_available = False
            import_utils._accelerate_available = False
            
            # Method 2: Also patch importlib.util.find_spec to handle our mocked modules
            original_find_spec = importlib.util.find_spec
            
            def patched_find_spec(name, package=None):
                if name in STANDARD_MOCK_MODULES or any(name.startswith(f"{mod}.") for mod in STANDARD_MOCK_MODULES):
                    # Return a dummy spec for our mocked modules
                    print(f"Using dummy spec for {name}")
                    return DummyModuleSpec(name)
                try:
                    return original_find_spec(name, package)
                except ValueError:
                    # Handle ValueError (e.g. for bitsandbytes.__spec__ is None)
                    if name in sys.modules:
                        module = sys.modules[name]
                        if hasattr(module, '__spec__'):
                            return module.__spec__
                    return None
                
            # Apply the importlib patch - this helps with find_spec issues
            importlib.util.find_spec = patched_find_spec
            print("Patched importlib.util.find_spec to handle mocked modules")
            
            return True
            
        except ImportError:
            print("transformers.utils.import_utils not found, unable to patch directly")
            
        # Method 3: Alternative approach if direct patching fails
        def patch_importlib():
            """Patch importlib to handle our mocked modules."""
            # Patch find_spec to avoid issues with mocked modules
            original_find_spec = importlib.util.find_spec
            
            def patched_find_spec(name, package=None):
                # For our mocked modules, return a custom spec
                if name in STANDARD_MOCK_MODULES or any(name.startswith(f"{mod}.") for mod in STANDARD_MOCK_MODULES):
                    return DummyModuleSpec(name)
                try:
                    return original_find_spec(name, package)
                except ValueError:
                    # Handle ValueError (likely from None specs)
                    return None
                except Exception as e:
                    print(f"Error in find_spec for {name}: {e}")
                    return None
            
            # Apply the patch
            importlib.util.find_spec = patched_find_spec
            print("Patched importlib.util.find_spec")
            
            return True
            
        # Apply importlib patch as a fallback
        return patch_importlib()
        
    except Exception as e:
        print(f"Warning: Could not patch transformers: {e}")
        return False

# Patch quantize.linear function to avoid bitsandbytes
def patch_moshi_quantize():
    """Patch moshi.utils.quantize.linear to avoid dependency on bitsandbytes."""
    try:
        from moshi.utils import quantize
        orig_linear = getattr(quantize, 'linear', None)
        
        def patched_linear(module, input_tensor, weight_name='weight', bias_name=None):
            """Patched linear function that doesn't use bitsandbytes."""
            weight = getattr(module, weight_name)
            # Standard linear operation
            output = input_tensor @ weight.t()
            if bias_name is not None and hasattr(module, bias_name):
                bias = getattr(module, bias_name)
                output = output + bias.unsqueeze(0).expand_as(output)
            return output
        
        # Apply patch if the original exists
        if orig_linear is not None:
            quantize.linear = patched_linear
            print("Patched moshi.utils.quantize.linear to avoid bitsandbytes dependency")
        return True
    except Exception as e:
        print(f"Warning: Could not patch moshi.utils.quantize: {e}")
        return False

# Fix import errors by installing mocks
def install_all_mocks():
    """Install all mocks and patches."""
    install_dummy_modules(STANDARD_MOCK_MODULES)
    patch_moshi_quantize() 
    patch_transformers()
    print("All mock modules and patches installed successfully")
    
if __name__ == "__main__":
    install_all_mocks()
