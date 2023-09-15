

def test_nos_imports():
    import nos


def test_nos_internal_imports():
    import nos
    
    # Try importing the internal module
    try:
        import autonomi.nos._internal
        success = True
    except ImportError:
        success = False
    
    assert success == nos._internal_available