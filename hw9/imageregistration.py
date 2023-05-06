import cv2
import numpy

def main():
    src = cv2.imread("Source.png")
    trg = cv2.imread("Target.png")
    warped_resize_array = []

    #part 1
    src_pts = numpy.array([[50, 50], [50, 200], [200, 200], [200, 50]])
    tgt_pts = numpy.array([[150, 100], [100, 300], [300, 300], [250, 100]])

    A = numpy.zeros((8, 9))
    for i in range(4):
        x, y = src_pts[i]
        u, v = tgt_pts[i]
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]
    
    ATA = A.T @ A
    ATb = A.T @ tgt_pts.reshape(-1)
    h = (numpy.linalg.inv(ATA) @ ATb).reshape((3, 3))

    warped = cv2.warpPerspective(src, h, (trg.shape[1], trg.shape[0]))

    #resize images to match
    warped = cv2.resize(warped, (trg.shape[1], trg.shape[0]))
    src = cv2.resize(src, (trg.shape[1], trg.shape[0]))

    #combine images
    combined = numpy.concatenate((src, trg, warped), 1)
    cv2.imshow("Part 1 - Combined image of source, target, and warped", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #part 2
    # Solve for the homography transformation matrix using eigen decomposition
    eigenvalues, eigenvectors = numpy.linalg.eig(ATA)
    smallest_eigenvalue_index = numpy.argmin(eigenvalues)
    h_eig = eigenvectors[:, smallest_eigenvalue_index]
    h_1 = numpy.reshape(h_eig, (3, 3))

    # Apply the homography transformation to the source image
    warped = cv2.warpPerspective(src, h_1, (trg.shape[1], trg.shape[0]))

    # Resize the images to have the same height
    height = max(src.shape[0], trg.shape[0], warped.shape[0])
    src_resized = cv2.resize(src, (int(src.shape[1]*height/src.shape[0]), height))
    trg_resized = cv2.resize(trg, (int(trg.shape[1]*height/trg.shape[0]), height))
    warped_resized = cv2.resize(warped, (int(warped.shape[1]*height/warped.shape[0]), height))
    warped_resize_array.append(warped_resized)

    # Display the source, target, and warped images side by side
    combined_img = numpy.concatenate((src_resized, trg_resized, warped_resized), 1)
    cv2.imshow('Part 2 - Combined image of source, target, and warped', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # part 3
    _, _, V = numpy.linalg.svd(A)
    h_svd = V[-1].reshape((3, 3))

    #apply the homography transformation to the src
    warped = cv2.warpPerspective(src, h_svd, (trg.shape[1], src.shape[0]))

    #resize images to same height
    height = max(src.shape[0], trg.shape[0], warped.shape[0])
    src_resized = cv2.resize(src, (int(src.shape[1]*height/src.shape[0]), height))
    trg_resized = cv2.resize(trg, (int(trg.shape[1]*height/trg.shape[0]), height))
    warped_resized = cv2.resize(warped, (int(warped.shape[1]*height/warped.shape[0]), height))
    warped_resize_array.append(warped_resized)

    #display
    combined_img = numpy.concatenate((src_resized, trg_resized, warped_resized), 1)
    cv2.imshow('Part 3 - Combined image of source, target, and warped', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #part 4
    combined = numpy.concatenate((warped_resize_array[0], warped_resize_array[1]), 1)
    cv2.putText(combined, "Images are similar: h_eig and h_svd are identical", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Part 4 - Comparing h_eig, h_svd', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # flag = numpy.allclose(h_eig, h, 0, 8)
    # print(flag)

    #part 5
    x, y = numpy.meshgrid(numpy.arange(trg.shape[1]), numpy.arange(trg.shape[0]))
    target = numpy.column_stack((x.flatten(), y.flatten()))

    #warp each point to source's corresponding location
    src_pts = cv2.perspectiveTransform(numpy.float32([target]), h_1)[0]

    #reshape the warped points to match target
    src_pts = src_pts.reshape((trg.shape[1], trg.shape[0], 2))

    #use remap function to resample the source image
    forward_warp = cv2.remap(src, src_pts, None, cv2.INTER_LINEAR)

    #resize the images to same height
    height = max(src.shape[0], trg.shape[0], forward_warp.shape[0])
    src_img_resized = cv2.resize(src, (int(src.shape[1]*height/src.shape[0]), height))
    trg_img_resized = cv2.resize(trg, (int(trg.shape[1]*height/trg.shape[0]), height))
    forward_warp_resized = cv2.resize(forward_warp, (int(forward_warp.shape[1]*height/forward_warp.shape[0]), height))

    #combine each picture and output
    combined_img = numpy.concatenate((src_img_resized, trg_img_resized, forward_warp_resized), axis=1)
    cv2.imshow('Part 5 - Combined image of source, target, and forward warped', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #part 5
    x, y = numpy.meshgrid(numpy.arange(trg.shape[1]), numpy.arange(trg.shape[0]))
    target = numpy.column_stack((x.flatten(), y.flatten()))

    #warp each point to source's corresponding location
    src_pts = cv2.perspectiveTransform(numpy.float32([target]), h_1)[0]

    #reshape the warped points
    src_pts = src_pts.reshape((trg.shape[1], trg.shape[0], 2))

    #resample the source image using the warped grid of points
    backward_warp = cv2.remap(src, src_pts, None, cv2.INTER_LINEAR)

    #resize the images to have the same height for output
    height = max(src.shape[0], trg.shape[0], backward_warp.shape[0])
    src_img_resized = cv2.resize(src, (int(src.shape[1]*height/src.shape[0]), height))
    trg_img_resized = cv2.resize(trg, (int(trg.shape[1]*height/trg.shape[0]), height))
    backward_warp_resize = cv2.resize(backward_warp, (int(backward_warp.shape[1]*height/backward_warp.shape[0]), height))

    #combine each picture together
    combined_img = numpy.concatenate((src_img_resized, trg_img_resized, backward_warp_resize), axis=1)
    cv2.imshow('Part 6 - Combined image of source, target, and backward warped', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()