prediction_result = get_prediction(
    image=image_list[0],
    detection_model=detection_model,
    shift_amount=shift_amount_list[0],
    full_shape=[
        slice_image_result.original_image_height,
        slice_image_result.original_image_width,
    ],
)