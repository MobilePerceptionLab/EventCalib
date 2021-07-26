1. cv_calib is the conventional camera calibration module.
2. dbscan is the DBSCAN cluster algorithm.
3. event is the base classes for events.
4. event_camera_calib is the proposed event camera calibration algorithm.


The usage of event_camera_calib:
1. Store the events into binary format described in modules/camera_calibration/event/tool/txt2bin.cpp.
2. Modify config file, parameter/event-calibration-1101-seq01.yaml.
2. Usage: unit_test_eventCameraCalib settingFilePath binFilePath SavePath