
# Virtual Try-On Web Application Setup Instructions

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Prepare Clothing Items
- Create a `clothing_items` folder in the project root
- Add clothing item images (PNG, JPG, JPEG) to this folder
- Images should be clear product photos with transparent or white backgrounds
- Recommended image size: 512x512 pixels

Example clothing items structure:
```
clothing_items/
├── red_tshirt.jpg
├── blue_jeans.png
├── black_jacket.jpg
├── white_dress.png
└── striped_shirt.jpg
```

## 3. Run the Application

### Option 1: Flask Web App (Recommended)
```bash
python app.py
```
Then open your browser and go to: http://localhost:5000

### Option 2: Command Line Demo
```bash
python virtual_tryon_main.py
```

## 4. Using the Web Application

1. **Start Camera**: Click "Start Camera" to begin video feed
2. **Capture Image**: Click "Capture" to take your photo
3. **Select Clothing**: Choose a clothing item from the grid
4. **Try On**: Click "Try On" to see the virtual fitting
5. **Additional Features**:
   - "Detect Pose": See pose landmarks detection
   - "Segment Body": View body segmentation

## 5. API Endpoints

- `GET /` - Main web application
- `GET /api/clothing-items` - Get available clothing items
- `POST /api/try-on` - Perform virtual try-on
- `POST /api/pose-detection` - Detect pose landmarks
- `POST /api/body-segmentation` - Segment body from background

## 6. Technical Architecture

The system uses:
- **MediaPipe**: For pose detection and body segmentation
- **OpenCV**: For image processing and computer vision
- **Flask**: For web application framework
- **TensorFlow**: For machine learning capabilities

## 7. Performance Optimization

For better performance:
- Use good lighting conditions
- Stand 2-3 feet from the camera
- Wear contrasting colors to background
- Ensure clear visibility of body pose

## 8. Troubleshooting

**Camera not working:**
- Ensure camera permissions are granted
- Try using HTTPS for better browser compatibility

**No clothing items showing:**
- Check that images are in the `clothing_items` folder
- Ensure images have proper extensions (.jpg, .png, .jpeg)

**Pose detection failing:**
- Ensure good lighting
- Stand fully visible in frame
- Try different poses or angles

## 9. Development and Customization

To customize the system:
- Modify `virtual_tryon_main.py` for core ML logic
- Edit `app.py` for web application features
- Update `templates/index.html` for UI changes
- Add new clothing datasets in the implementation plan

## 10. Next Steps

- Add more sophisticated garment warping algorithms
- Implement real-time video processing
- Add clothing recommendation features
- Integrate with e-commerce platforms
- Deploy to cloud platforms (AWS, GCP, Azure)
