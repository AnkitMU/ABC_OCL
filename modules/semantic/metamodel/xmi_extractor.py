"""
XMI Metamodel Extractor
Integrates with the existing verification framework's XMI parser
"""

import sys
from pathlib import Path
from typing import Optional

# Add verification framework to path
# Try multiple possible locations
VERIFICATION_FRAMEWORK_PATHS = [
    Path(__file__).parent.parent.parent.parent / 'hybrid-ssr-ocl-full-extended' / 'src',  # Local copy (preferred)
    Path('/Users/ankitjha/Downloads/hybrid-ssr-ocl-full-extended/src'),  # Fallback: original location
]

# Add first valid path
for fpath in VERIFICATION_FRAMEWORK_PATHS:
    if fpath.exists():
        sys.path.insert(0, str(fpath))
        break

try:
    from ssr_ocl.lowering.association_backed_encoder import XMIMetadataExtractor as VerificationXMIExtractor
    from ssr_ocl.lowering.association_backed_encoder import AssociationMetadata, AttributeMetadata
    HAS_VERIFICATION_FRAMEWORK = True
except ImportError:
    HAS_VERIFICATION_FRAMEWORK = False
    # Verification framework not found - using standalone XML parser

# Import our core models
from modules.core.models import Metamodel, Class, Attribute, Association


class MetamodelExtractor:
    """
    Extracts metamodel from XMI files
    
    Uses the existing verification framework's parser when available,
    falls back to standalone XML parsing otherwise.
    """
    
    def __init__(self, xmi_file: str):
        """
        Initialize extractor
        
        Args:
            xmi_file: Path to XMI file
        """
        self.xmi_file = xmi_file
        self.metamodel: Optional[Metamodel] = None
        self._extract()
    
    def _extract(self):
        """Extract metamodel from XMI"""
        if HAS_VERIFICATION_FRAMEWORK:
            self._extract_with_verification_framework()
        else:
            self._extract_with_standalone_parser()
    
    def _extract_with_verification_framework(self):
        """Extract using verification framework's XMI parser"""
        # Loading metamodel from XMI (using verification framework)
        
        # Use existing parser
        extractor = VerificationXMIExtractor(self.xmi_file)
        
        # Convert to our metamodel format
        metamodel = Metamodel()
        
        # Create classes
        for class_name in extractor.classes:
            cls = Class(name=class_name)
            
            # Add attributes
            for attr_meta in extractor.get_attributes_for_class(class_name):
                attr = Attribute(
                    name=attr_meta.attr_name,
                    type=attr_meta.attr_type,
                    lower=1,  # Default - could be enhanced
                    upper=1
                )
                cls.attributes.append(attr)
            
            # Add associations
            for assoc_meta in extractor.get_associations_for_class(class_name):
                # Determine multiplicity
                lower = assoc_meta.lower_bound
                upper = assoc_meta.upper_bound if assoc_meta.upper_bound is not None else -1
                
                assoc = Association(
                    name=assoc_meta.ref_name,  # Use ref_name directly (no class prefix)
                    ref_name=assoc_meta.ref_name,  # Role name
                    source_class=assoc_meta.source_class,
                    target_class=assoc_meta.target_class,
                    lower=lower,
                    upper=upper,
                    is_composition=assoc_meta.containment,
                    is_bidirectional=bool(assoc_meta.opposite_ref),
                    opposite_ref=assoc_meta.opposite_ref if assoc_meta.opposite_ref else None
                )
                cls.associations.append(assoc)
            
            metamodel.add_class(cls)
        
        self.metamodel = metamodel
        # Loaded metamodel summary (suppressed)
    
    def _extract_with_standalone_parser(self):
        """Fallback: standalone XML parser"""
        import xml.etree.ElementTree as ET
        
        # Loading metamodel from XMI (standalone parser)
        
        try:
            tree = ET.parse(self.xmi_file)
            root = tree.getroot()
            
            metamodel = Metamodel()
            
            # Find all classes
            for elem in root.findall(".//{http://www.eclipse.org/emf/2002/Ecore}EClass"):
                class_name = elem.get('name')
                if not class_name:
                    continue
                
                cls = Class(
                    name=class_name,
                    is_abstract=elem.get('abstract', 'false') == 'true'
                )
                
                # Extract attributes
                for attr_elem in elem.findall(".//{http://www.eclipse.org/emf/2002/Ecore}EAttribute"):
                    attr_name = attr_elem.get('name')
                    attr_type = attr_elem.get('eType', 'String')
                    
                    if attr_name:
                        # Parse type
                        if '#//' in attr_type:
                            attr_type = attr_type.split('#//')[-1]
                        
                        lower_str = attr_elem.get('lowerBound', '1')
                        upper_str = attr_elem.get('upperBound', '1')
                        
                        lower = int(lower_str) if lower_str != '-1' else 0
                        upper = int(upper_str) if upper_str != '-1' else -1
                        
                        attr = Attribute(
                            name=attr_name,
                            type=attr_type,
                            lower=lower,
                            upper=upper
                        )
                        cls.attributes.append(attr)
                
                # Extract associations (references)
                for ref_elem in elem.findall(".//{http://www.eclipse.org/emf/2002/Ecore}EReference"):
                    ref_name = ref_elem.get('name')
                    target_type = ref_elem.get('eType', '')
                    
                    if ref_name and target_type:
                        # Parse target class
                        if '#//' in target_type:
                            target_class = target_type.split('#//')[-1]
                        else:
                            target_class = target_type
                        
                        lower_str = ref_elem.get('lowerBound', '0')
                        upper_str = ref_elem.get('upperBound', '-1')
                        containment = ref_elem.get('containment', 'false') == 'true'
                        opposite = ref_elem.get('eOpposite', '')
                        
                        lower = int(lower_str) if lower_str != '-1' else 0
                        upper = int(upper_str) if upper_str != '-1' else -1
                        
                        assoc = Association(
                            name=ref_name,  # Use ref_name directly (no class prefix)
                            ref_name=ref_name,
                            source_class=class_name,
                            target_class=target_class,
                            lower=lower,
                            upper=upper,
                            is_composition=containment,
                            is_bidirectional=bool(opposite),
                            opposite_ref=opposite if opposite else None
                        )
                        cls.associations.append(assoc)
                
                metamodel.add_class(cls)
            
            self.metamodel = metamodel
            # Loaded metamodel summary (suppressed)
            
        except Exception as e:
            print(f"Error parsing XMI: {e}")
            raise
    
    def get_metamodel(self) -> Metamodel:
        """Get the extracted metamodel"""
        if self.metamodel is None:
            raise ValueError("Metamodel not extracted")
        return self.metamodel
    
    def print_summary(self):
        """Print a summary of the extracted metamodel"""
        if not self.metamodel:
            print("No metamodel loaded")
            return
        
        print("\n" + "="*60)
        print("METAMODEL SUMMARY")
        print("="*60)
        
        for class_name in sorted(self.metamodel.get_class_names()):
            cls = self.metamodel.get_class(class_name)
            print(f"\n {cls}")
            
            if cls.attributes:
                print(f"   Attributes:")
                for attr in cls.attributes:
                    print(f"      • {attr}")
            
            if cls.associations:
                print(f"   Associations:")
                for assoc in cls.associations:
                    print(f"      • {assoc}")
        
        print("\n" + "="*60)


# Convenience function
def extract_metamodel(xmi_file: str) -> Metamodel:
    """
    Extract metamodel from XMI file
    
    Args:
        xmi_file: Path to XMI file
        
    Returns:
        Extracted metamodel
    """
    extractor = MetamodelExtractor(xmi_file)
    return extractor.get_metamodel()


# Test function
def main():
    """Test the extractor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python xmi_extractor.py <xmi_file>")
        return
    
    xmi_file = sys.argv[1]
    
    try:
        extractor = MetamodelExtractor(xmi_file)
        extractor.print_summary()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
