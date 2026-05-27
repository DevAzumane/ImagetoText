import cv2


def sort_contours(contours, method="left-to-right"):

    reverse = False
    i = 0

    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True

    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1

    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    combined = zip(contours, bounding_boxes)

    sorted_combined = sorted(
        combined,
        key=lambda b: b[1][i],
        reverse=reverse
    )

    contours, bounding_boxes = zip(*sorted_combined)

    return contours