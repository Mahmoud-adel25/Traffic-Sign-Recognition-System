# Traffic Sign Classes Mapping
# GTSRB (German Traffic Sign Recognition Benchmark) Dataset

TRAFFIC_SIGN_CLASSES = {
    0: {
        'name': 'Speed limit 20 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 20 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    1: {
        'name': 'Speed limit 30 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 30 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    2: {
        'name': 'Speed limit 50 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 50 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    3: {
        'name': 'Speed limit 60 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 60 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    4: {
        'name': 'Speed limit 70 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 70 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    5: {
        'name': 'Speed limit 80 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 80 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    6: {
        'name': 'End of speed limit 80 km/h',
        'category': 'Speed Limit',
        'description': 'End of 80 km/h speed limit zone',
        'color': 'White circle with black border',
        'shape': 'Circular'
    },
    7: {
        'name': 'Speed limit 100 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 100 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    8: {
        'name': 'Speed limit 120 km/h',
        'category': 'Speed Limit',
        'description': 'Maximum speed limit of 120 kilometers per hour',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    9: {
        'name': 'No passing',
        'category': 'Prohibition',
        'description': 'No overtaking allowed for vehicles',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    10: {
        'name': 'No passing for vehicles over 3.5 metric tons',
        'category': 'Prohibition',
        'description': 'No overtaking allowed for heavy vehicles',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    11: {
        'name': 'Right-of-way at the next intersection',
        'category': 'Priority',
        'description': 'You have right-of-way at the next intersection',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    12: {
        'name': 'Priority road',
        'category': 'Priority',
        'description': 'You are on a priority road',
        'color': 'Yellow diamond with white border',
        'shape': 'Diamond'
    },
    13: {
        'name': 'Yield',
        'category': 'Priority',
        'description': 'Yield to other vehicles',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    14: {
        'name': 'Stop',
        'category': 'Priority',
        'description': 'Stop and give way to all vehicles',
        'color': 'Red octagon with white text',
        'shape': 'Octagonal'
    },
    15: {
        'name': 'No vehicles',
        'category': 'Prohibition',
        'description': 'No vehicles allowed',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    16: {
        'name': 'Vehicles over 3.5 metric tons prohibited',
        'category': 'Prohibition',
        'description': 'Heavy vehicles not allowed',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    17: {
        'name': 'No entry',
        'category': 'Prohibition',
        'description': 'No entry for all vehicles',
        'color': 'Red circle with white background',
        'shape': 'Circular'
    },
    18: {
        'name': 'General caution',
        'category': 'Warning',
        'description': 'General warning sign',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    19: {
        'name': 'Dangerous curve left',
        'category': 'Warning',
        'description': 'Dangerous curve to the left ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    20: {
        'name': 'Dangerous curve right',
        'category': 'Warning',
        'description': 'Dangerous curve to the right ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    21: {
        'name': 'Double curve',
        'category': 'Warning',
        'description': 'Double curve ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    22: {
        'name': 'Bumpy road',
        'category': 'Warning',
        'description': 'Bumpy or uneven road surface',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    23: {
        'name': 'Slippery road',
        'category': 'Warning',
        'description': 'Road surface may be slippery',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    24: {
        'name': 'Road narrows on the right',
        'category': 'Warning',
        'description': 'Road narrows on the right side',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    25: {
        'name': 'Road work',
        'category': 'Warning',
        'description': 'Road work or construction ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    26: {
        'name': 'Traffic signals',
        'category': 'Warning',
        'description': 'Traffic light signals ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    27: {
        'name': 'Pedestrians',
        'category': 'Warning',
        'description': 'Pedestrian crossing ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    28: {
        'name': 'Children crossing',
        'category': 'Warning',
        'description': 'Children crossing ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    29: {
        'name': 'Bicycles crossing',
        'category': 'Warning',
        'description': 'Bicycle crossing ahead',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    30: {
        'name': 'Snow/ice',
        'category': 'Warning',
        'description': 'Snow or ice on road surface',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    31: {
        'name': 'Wild animals crossing',
        'category': 'Warning',
        'description': 'Wild animals may cross the road',
        'color': 'White triangle with red border',
        'shape': 'Triangular'
    },
    32: {
        'name': 'End of all speed and passing limits',
        'category': 'Speed Limit',
        'description': 'End of all speed and passing restrictions',
        'color': 'White circle with black border',
        'shape': 'Circular'
    },
    33: {
        'name': 'Turn right ahead',
        'category': 'Mandatory',
        'description': 'Turn right at the next intersection',
        'color': 'Blue circle with white arrow',
        'shape': 'Circular'
    },
    34: {
        'name': 'Turn left ahead',
        'category': 'Mandatory',
        'description': 'Turn left at the next intersection',
        'color': 'Blue circle with white arrow',
        'shape': 'Circular'
    },
    35: {
        'name': 'Ahead only',
        'category': 'Mandatory',
        'description': 'Continue straight ahead only',
        'color': 'Blue circle with white arrow',
        'shape': 'Circular'
    },
    36: {
        'name': 'Go straight or turn right',
        'category': 'Mandatory',
        'description': 'Go straight or turn right',
        'color': 'Blue circle with white arrows',
        'shape': 'Circular'
    },
    37: {
        'name': 'Go straight or turn left',
        'category': 'Mandatory',
        'description': 'Go straight or turn left',
        'color': 'Blue circle with white arrows',
        'shape': 'Circular'
    },
    38: {
        'name': 'Keep right',
        'category': 'Mandatory',
        'description': 'Keep to the right side of the road',
        'color': 'Blue circle with white arrow',
        'shape': 'Circular'
    },
    39: {
        'name': 'Keep left',
        'category': 'Mandatory',
        'description': 'Keep to the left side of the road',
        'color': 'Blue circle with white arrow',
        'shape': 'Circular'
    },
    40: {
        'name': 'Roundabout mandatory',
        'category': 'Mandatory',
        'description': 'Roundabout ahead, follow the direction',
        'color': 'Blue circle with white arrows',
        'shape': 'Circular'
    },
    41: {
        'name': 'End of no passing',
        'category': 'Prohibition',
        'description': 'End of no passing zone',
        'color': 'White circle with black border',
        'shape': 'Circular'
    },
    42: {
        'name': 'End of no passing by vehicles over 3.5 metric tons',
        'category': 'Prohibition',
        'description': 'End of no passing zone for heavy vehicles',
        'color': 'White circle with black border',
        'shape': 'Circular'
    }
}

# Category colors for visualization
CATEGORY_COLORS = {
    'Speed Limit': '#FF6B6B',      # Red
    'Warning': '#FFA500',          # orange
    'Prohibition': '#4ECDC4',      # Teal
    'Priority': '#45B7D1',         # Blue
    'Mandatory': '#96CEB4'         # Green 
}

# Category descriptions
CATEGORY_DESCRIPTIONS = {
    'Speed Limit': 'Regulatory signs that indicate maximum speed limits',
    'Warning': 'Warning signs that alert drivers to potential hazards',
    'Prohibition': 'Regulatory signs that prohibit certain actions',
    'Priority': 'Signs that indicate right-of-way rules',
    'Mandatory': 'Regulatory signs that require specific actions'
}

def get_class_info(class_id):
    """Get information about a specific traffic sign class"""
    return TRAFFIC_SIGN_CLASSES.get(class_id, {
        'name': f'Unknown Class {class_id}',
        'category': 'Unknown',
        'description': 'Unknown traffic sign class',
        'color': 'Unknown',
        'shape': 'Unknown'
    })

def get_class_names():
    """Get list of all class names"""
    return [TRAFFIC_SIGN_CLASSES[i]['name'] for i in range(len(TRAFFIC_SIGN_CLASSES))]

def get_categories():
    """Get list of all categories"""
    return list(CATEGORY_COLORS.keys())

def get_classes_by_category(category):
    """Get all classes in a specific category"""
    return [class_id for class_id, info in TRAFFIC_SIGN_CLASSES.items() 
            if info['category'] == category]
