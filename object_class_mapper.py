"""
Object Class Mapper for Basketball Shot Detection
Handles mapping of different model output labels to unified object types
"""

class ObjectClassMapper:
    """
    Maps various object detection model labels to unified object types.
    This allows different models with different class naming conventions to be used consistently.
    """
    
    def __init__(self):
        # Define the mapping from various class names to unified object types
        self.object_class_mapping = {
            'ball': [
                'Ball', 'Basketball', 'Sports ball', 'Sport ball', 'ball', 
                'basketball', 'sports ball', 'sport ball'
            ],
            'hoop': [
                'Basketball Hoop', 'Hoop', 'Rim', 'basketball hoop', 
                'hoop', 'rim'
            ],
            'person': [
                'Person', 'Human', 'person', 'human'
            ]
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_mapping = {}
        for obj_type, class_names in self.object_class_mapping.items():
            for class_name in class_names:
                self.reverse_mapping[class_name.lower()] = obj_type

    def get_object_type(self, class_name):
        """
        Get the unified object type for a given class name.
        
        Args:
            class_name (str): The class name from model detection
            
        Returns:
            str or None: The unified object type ('ball', 'hoop', 'person') or None if not found
        """
        normalized_name = class_name.strip().lower()
        return self.reverse_mapping.get(normalized_name, None)
        
    def is_object_type(self, class_name, object_type):
        """
        Check if a class name corresponds to a specific object type.
        
        Args:
            class_name (str): The class name from model detection
            object_type (str): The target object type ('ball', 'hoop', 'person')
            
        Returns:
            bool: True if the class name matches the object type, False otherwise
        """
        return self.get_object_type(class_name) == object_type
        
    def get_all_object_types(self):
        """
        Get all supported object types.
        
        Returns:
            list: List of all supported object types
        """
        return list(self.object_class_mapping.keys())
        
    def get_class_names_for_type(self, object_type):
        """
        Get all class names that map to a specific object type.
        
        Args:
            object_type (str): The object type ('ball', 'hoop', 'person')
            
        Returns:
            list: List of class names that map to the object type
        """
        return self.object_class_mapping.get(object_type, [])
        

# Global instance for easy access throughout the project
object_class_mapper = ObjectClassMapper()