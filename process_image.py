def processImage(opWrapper, imageToProcess, output_path, heatmap_path):
    # Process Image
    datum = op.Datum()    
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # Display Image
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imwrite(output_path, datum.cvOutputData)

    # Process outputs
    outputImageF = (datum.inputNetData[0].copy())[0,:,:,:] + 0.5
    outputImageF = cv2.merge([outputImageF[0,:,:], outputImageF[1,:,:], outputImageF[2,:,:]])
    outputImageF = (outputImageF*255.).astype(dtype='uint8')
    heatmaps = datum.poseHeatMaps.copy()
    heatmaps = (heatmaps).astype(dtype='uint8')

    # Display Image
    num_maps = heatmaps.shape[0]
    for counter in range(num_maps):
        heatmap = heatmaps[counter, :, :].copy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        cv2.imwrite(heatmap_path + "heatmap_"+ str(counter).zfill(2) + ".jpg", heatmap)