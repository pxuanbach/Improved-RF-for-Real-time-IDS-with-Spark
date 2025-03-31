from utils import load_checkpoint, save_checkpoint, clear_checkpoint, bcolors

def test_checkpoint_functions():
    print("🧪 Testing checkpoint functions...")
    
    # Test data
    test_features = [
        (0, ["feature1", "feature2"], [0.5, 0.7]),
        (1, ["feature3", "feature4"], [0.6, 0.8])
    ]
    
    # Test 1: Save checkpoint
    print("\n📝 Test 1: Saving checkpoint...")
    save_checkpoint(1, test_features)
    
    # Test 2: Load checkpoint
    print("\n📖 Test 2: Loading checkpoint...")
    last_split, loaded_features = load_checkpoint()
    assert last_split == 1, f"Expected last_split to be 1, got {last_split}"
    assert len(loaded_features) == 2, f"Expected 2 features, got {len(loaded_features)}"
    print("✅ Loaded checkpoint matches saved data")
    
    # Test 3: Clear checkpoint
    print("\n🗑️ Test 3: Clearing checkpoint...")
    clear_checkpoint()
    last_split, loaded_features = load_checkpoint()
    assert last_split == -1, f"Expected last_split to be -1 after clearing, got {last_split}"
    assert len(loaded_features) == 0, f"Expected empty features after clearing, got {len(loaded_features)}"
    print("✅ Checkpoint cleared successfully")
    
    # Test 4: Simulate interrupted processing and resume
    print("\n⏸️ Test 4: Testing resume functionality...")
    
    # Save initial state
    initial_features = [(0, ["feature1"], [0.5])]
    save_checkpoint(0, initial_features)
    
    # Simulate loading and resuming
    last_split, features = load_checkpoint()
    assert last_split == 0, "Failed to load initial state"
    
    # Simulate adding more data after resume
    features.append((1, ["feature2"], [0.7]))
    save_checkpoint(1, features)
    
    # Verify final state
    last_split, final_features = load_checkpoint()
    assert last_split == 1, "Failed to save resumed state"
    assert len(final_features) == 2, "Failed to append new data after resume"
    
    print("✅ Resume functionality working correctly")
    
    # Cleanup
    # clear_checkpoint()
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    try:
        test_checkpoint_functions()
    except AssertionError as e:
        print(f"❌ Test failed: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
