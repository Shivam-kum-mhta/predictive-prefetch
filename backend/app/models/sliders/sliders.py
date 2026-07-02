# Re orders recommendations 
# based on model config sliders (MODEL_SETTINGS.toml)

import toml
import os
from .base_slider_class import Slider
from .popularity_bias import PopularityBias
from .recency_bias import RecencyBias
from .title_correlation_bias import TitleCorrelationBias
from .category_bias import CategoryBias
from .subcategory_bias import SubcategoryBias


# Map class names to actual classes
SLIDER_CLASSES = {
    "RecencyBias": RecencyBias,
    "PopularityBias": PopularityBias,
    "TitleCorrelationBias": TitleCorrelationBias,
    "CategoryBias": CategoryBias,
    "SubcategoryBias": SubcategoryBias,
}


def get_sliders_config():
    """Get the sliders configuration from MODEL_SETTINGS.toml"""
    settings_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MODEL_SETTINGS.toml')
    with open(settings_path, "r") as f:
        settings = toml.load(f)
    return settings["sliders"]


def reorder_recommendations(recommendations, history):
    """
    Reorder recommendations based on the model config sliders (MODEL_SETTINGS.toml)
    
    Args:
        recommendations: List of dicts with 'article_id' and 'probability'
        history: List of article IDs in user history
        
    Returns:
        Reordered list of recommendations
    """
    sliders_config = get_sliders_config()
    
    # Extract the base config (cur_date_fake, etc.)
    base_config = {k: v for k, v in sliders_config.items() if not isinstance(v, dict)}
    
    # Iterate through each slider configuration
    for slider_name, slider_config in sliders_config.items():
        # Skip non-dict entries (like cur_date_fake)
        if not isinstance(slider_config, dict):
            continue

        # print(f"Slider config: {slider_config}")
        
        # Check if slider is enabled
        if slider_config.get("enable", False):
            class_name = slider_config.get("class")
            
            if class_name not in SLIDER_CLASSES:
                print(f"Warning: Unknown slider class '{class_name}', skipping...")
                continue
            
            # Merge base config with slider-specific config
            merged_config = {**base_config, **slider_config}
            
            # Instantiate and apply the slider
            try:
                base_slider_class = SLIDER_CLASSES[class_name]
                slider_instance = base_slider_class(slider_name, merged_config, history)
                recommendations = slider_instance.apply(recommendations)
            except Exception as e:
                print(f"Error applying slider '{slider_name}': {e}")
                # Continue with other sliders even if one fails
    # print(f"Recommendations::::", len(recommendations))
    return recommendations


