#!/usr/bin/env python3
"""
Example usage of the Caption Manager without VLM model.
This demonstrates the data persistence and management features.
"""

import os
import tempfile
from pathlib import Path
from caption_tool import CaptionManager


def main():
    """Demonstrate caption management features."""
    print("🖼️  Caption Manager Demo\n")
    
    # Create a temporary directory for the demo
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temporary directory: {tmpdir}\n")
        
        # Create some mock image files
        image_files = []
        for i in range(1, 4):
            img_path = os.path.join(tmpdir, f"sample_image_{i}.jpg")
            Path(img_path).touch()
            image_files.append(img_path)
            print(f"Created: {Path(img_path).name}")
        
        print("\n" + "="*60)
        
        # Initialize caption manager
        manager = CaptionManager()
        
        # Add sample captions
        captions = [
            "A serene landscape with mountains and a clear blue sky",
            "A close-up portrait of a smiling person wearing sunglasses",
            "An abstract pattern with vibrant colors and geometric shapes"
        ]
        
        print("\nAdding captions...")
        for img_path, caption in zip(image_files, captions):
            manager.add_caption(img_path, caption)
            print(f"✓ Added caption for {Path(img_path).name}")
        
        print("\n" + "="*60)
        
        # Retrieve a caption
        print("\nRetrieving caption for first image:")
        retrieved_caption = manager.get_caption(image_files[0])
        print(f"Caption: {retrieved_caption}")
        
        print("\n" + "="*60)
        
        # Save captions to text files
        print("\nSaving captions to text files...")
        result = manager.save_captions()
        print(f"✓ {result}")
        
        # Verify text files were created
        print("\nVerifying saved files:")
        for img_path in image_files:
            txt_path = Path(img_path).with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    content = f.read()
                print(f"✓ {txt_path.name} (length: {len(content)} chars)")
        
        print("\n" + "="*60)
        
        # Export metadata
        print("\nExporting metadata to JSON...")
        metadata_path = os.path.join(tmpdir, "metadata.json")
        result = manager.export_metadata(metadata_path)
        print(f"✓ {result}")
        
        # Show metadata content
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\nMetadata summary:")
        print(f"  - Total images: {metadata['image_count']}")
        print(f"  - Captions stored: {len(metadata['captions'])}")
        
        print("\n" + "="*60)
        print("\n✅ All operations completed successfully!")
        print("\nIn a real scenario with the VLM model:")
        print("  1. Run: python caption_tool.py")
        print("  2. Open the web interface")
        print("  3. Upload images")
        print("  4. Generate captions using the VLM")
        print("  5. Edit and refine as needed")
        print("  6. Save to disk for LoRA training")


if __name__ == "__main__":
    main()
