from hand_eye_calibration import HandEyeCalibrator



calibrator = HandEyeCalibrator(camera, robot, data_dir='./data', save_images=True)

T_base_cam = calibrator.calibrate(marker_length=0.07, num_samples=10, filter=True)

