"""
shape_analysis.py

Core, UI-independent erythrocyte shape analysis logic, extracted from app.py
so it can be unit tested in isolation (importing app.py directly would also
execute Streamlit UI calls and a live PDF download).
"""

import cv2
import numpy as np


def get_erythrocyte_shape_factors(image, anomaly_threshold, min_axis_size):
    """
    Detects erythrocytes in a BGR image via Otsu thresholding + contour/ellipse
    fitting, and computes per-cell shape metrics.

    Args:
        image: BGR image as a numpy array (as returned by cv2.imdecode/cv2.imread).
        anomaly_threshold: Shape Factor value above which a cell is classified
            as an anomaly (also drives the color-band cutoffs below it).
        min_axis_size: minimum major/minor axis length (px) required to keep
            a detected contour, used to filter out small segmentation artifacts.

    Returns:
        (processed_image, shape_factors_data, anomalies_data, binary_mask)
        - processed_image: copy of `image` annotated with fitted ellipses/axes.
        - shape_factors_data: list of dicts, one per normal/moderate cell
          (Shape Factor <= anomaly_threshold).
        - anomalies_data: list of dicts, one per anomalous cell
          (Shape Factor > anomaly_threshold).
        - binary_mask: the Otsu-thresholded binary image used for detection.
    """
    processed_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # IMPROVEMENT: Using Otsu's thresholding for automatic selection of the optimal threshold
    # This provides better segmentation in case of uneven illumination.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_factors_data = []
    anomalies_data = []

    for contour in contours:
        if len(contour) > 5:
            try:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse

                minor_axis = min(axes)
                major_axis = max(axes)

                # Condition for minimum size (artifact filtering)
                if major_axis < min_axis_size or minor_axis < min_axis_size:
                    continue

                if minor_axis > 0:
                    shape_factor = major_axis / minor_axis

                    # Calculating Ellipticity
                    ellipticity = 1 - (minor_axis / major_axis)

                    # Color bands are derived from the anomaly threshold itself
                    # (not fixed absolute values), so a cell's color on the image
                    # can never contradict its Normal/Anomaly classification in the
                    # table below, whatever threshold the user picks on the slider.
                    yellow_cutoff = anomaly_threshold * 0.75
                    red_cutoff = anomaly_threshold * 0.9

                    ellipse_color = (0, 255, 0)  # Green = normal

                    if shape_factor > yellow_cutoff and shape_factor <= red_cutoff:
                        ellipse_color = (0, 255, 255)  # Yellow = moderately elongated
                    elif shape_factor > red_cutoff:
                        ellipse_color = (0, 0, 255)  # Red = highly elongated (still below the anomaly threshold)

                    if shape_factor <= anomaly_threshold:
                        # Storing all metrics
                        shape_factors_data.append({
                            "Erythrocyte Number": len(shape_factors_data) + 1,
                            "Shape Factor": shape_factor,
                            "Ellipticity": ellipticity,
                            "Major Axis": major_axis,
                            "Minor Axis": minor_axis,
                            "Area": area,
                            "Perimeter": perimeter
                        })
                        cv2.ellipse(processed_image, ellipse, ellipse_color, 2)
                        angle_rad = np.radians(orientation)

                        # Drawing Axes (Major/Minor)
                        major_end_point_1 = (int(center[0] + major_axis/2 * np.cos(angle_rad)),
                                             int(center[1] + major_axis/2 * np.sin(angle_rad)))
                        major_end_point_2 = (int(center[0] - major_axis/2 * np.cos(angle_rad)),
                                             int(center[1] - major_axis/2 * np.sin(angle_rad)))
                        cv2.line(processed_image, major_end_point_1, major_end_point_2, (0, 0, 255), 1)

                        minor_angle_rad = np.radians(orientation + 90)
                        minor_end_point_1 = (int(center[0] + minor_axis/2 * np.cos(minor_angle_rad)),
                                             int(center[1] + minor_axis/2 * np.sin(minor_angle_rad)))
                        minor_end_point_2 = (int(center[0] - minor_axis/2 * np.cos(minor_angle_rad)),
                                             int(center[1] - minor_axis/2 * np.sin(minor_angle_rad)))
                        cv2.line(processed_image, minor_end_point_1, minor_end_point_2, (255, 0, 0), 1)

                        # Labeling the cell
                        cv2.putText(processed_image, str(len(shape_factors_data)),
                                    (int(center[0]) + 15, int(center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Storing all metrics for anomalies
                        anomalies_data.append({
                            "Erythrocyte Number": len(anomalies_data) + 1,
                            "Shape Factor": shape_factor,
                            "Ellipticity": ellipticity,
                            "Major Axis": major_axis,
                            "Minor Axis": minor_axis,
                            "Area": area,
                            "Perimeter": perimeter
                        })
                        cv2.ellipse(processed_image, ellipse, (255, 0, 255), 2)  # Magenta color for anomalies
                        cv2.putText(processed_image, f"A{len(anomalies_data)}",
                                    (int(center[0]) + 15, int(center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            except cv2.error:
                continue

    # Changed return values: added binary mask for diagnostic preview
    return processed_image, shape_factors_data, anomalies_data, thresh
