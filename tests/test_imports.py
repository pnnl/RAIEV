"""
Test import of raiev modules
"""

import atexit


def test_imports():
    """
    Check import of base raiev modules. Test will fail if import fails.
    """
    import raiev
    import raiev.models
    import raiev.models.base
    import raiev.models.hflm
    import raiev.models.hflmBase
    import raiev.utils
    import raiev.vis
    import raiev.workflows.accountability
    import raiev.workflows.equity
    import raiev.workflows.failurecost
    import raiev.workflows.transparency

    print("Tested imports of base raiev modules.")


atexit.register(test_imports)  # Print output after test.
