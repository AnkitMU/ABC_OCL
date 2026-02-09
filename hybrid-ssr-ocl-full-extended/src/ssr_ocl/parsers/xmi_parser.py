import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

def parse_xmi_model(model_path: str) -> Dict:
    """Parse XMI model to extract class information and relationships"""
    try:
        tree = ET.parse(model_path)
        root = tree.getroot()
        
        model_info = {
            'classes': {},
            'enums': {},
            'associations': {}
        }
        
        # Extract classes and their attributes
        for classifier in root.findall('.//eClassifiers'):
            class_type = classifier.get('{http://www.w3.org/2001/XMLSchema-instance}type')
            
            if class_type == 'ecore:EClass':
                class_name = classifier.get('name')
                if class_name:
                    model_info['classes'][class_name] = {
                        'attributes': [],
                        'references': []
                    }
                    
                    # Extract attributes and references
                    for feature in classifier.findall('.//eStructuralFeatures'):
                        feature_type = feature.get('{http://www.w3.org/2001/XMLSchema-instance}type')
                        feature_name = feature.get('name')
                        
                        if feature_type == 'ecore:EAttribute' and feature_name:
                            model_info['classes'][class_name]['attributes'].append({
                                'name': feature_name,
                                'type': feature.get('eType', '').split('#//')[-1]
                            })
                        elif feature_type == 'ecore:EReference' and feature_name:
                            target_type = feature.get('eType', '').split('#//')[-1]
                            upper_bound = feature.get('upperBound', '1')
                            is_collection = upper_bound == '-1'
                            
                            model_info['classes'][class_name]['references'].append({
                                'name': feature_name,
                                'type': target_type,
                                'is_collection': is_collection,
                                'containment': feature.get('containment', 'false') == 'true'
                            })
            
            elif class_type == 'ecore:EEnum':
                enum_name = classifier.get('name')
                if enum_name:
                    literals = []
                    for literal in classifier.findall('.//eLiterals'):
                        literal_name = literal.get('name')
                        if literal_name:
                            literals.append(literal_name)
                    model_info['enums'][enum_name] = literals
        
        return model_info
    
    except ET.ParseError as e:
        print(f"Error parsing XMI model: {e}")
        return {'classes': {}, 'enums': {}, 'associations': {}}

def extract_collection_mappings(model_info: Dict) -> Dict[str, str]:
    """Extract collection name to class name mappings from model"""
    mappings = {}
    
    for class_name, class_data in model_info['classes'].items():
        for ref in class_data['references']:
            if ref['is_collection']:
                collection_name = ref['name']
                target_class = ref['type']
                mappings[f"{collection_name}->"] = (target_class, collection_name)
    
    return mappings

def suggest_scope_bounds(model_info: Dict) -> Dict[str, int]:
    """Suggest reasonable scope bounds based on model structure"""
    bounds = {}
    
    for class_name, class_data in model_info['classes'].items():
        # Heuristic: classes with more references likely need larger scopes
        ref_count = len(class_data['references'])
        collection_count = sum(1 for ref in class_data['references'] if ref['is_collection'])
        
        if collection_count > 3:
            bounds[class_name] = 20  # Large collections
        elif collection_count > 1:
            bounds[class_name] = 10  # Medium collections  
        elif ref_count > 2:
            bounds[class_name] = 5   # Multiple references
        else:
            bounds[class_name] = 3   # Simple classes
    
    return bounds