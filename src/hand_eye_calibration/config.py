# serial = "234222302792" # old
# serial = "234222302792" # old
serial = "346522075401" # main camera
# serial = "244622072715" # side camera

# see readme for full configurations.
camera_config = {
    "enable_viewer": True,
    "enable_recorder": False,

    "specifications": {
        "fps": 30,
    "color_auto_exposure": False,
    "depth_auto_exposure": False,
    },

    "viewer": {                     # no need to keep this dict if "enable_viewer" is False
        "show_color": True,
        "show_depth": True,
        "fps": 10
    },

    "recorder": {                   # no need to keep this dict if "enable_recorder" is False
        "save_dir": "./recordings",
        "save_name": "test_session",
        "fps": 10,
        "save_with_overlays": True,
        "auto_start": False         # if False, press 's' to start recording at any time point
    }
}

