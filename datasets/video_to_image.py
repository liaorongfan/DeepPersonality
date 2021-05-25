"""
Extract all the 6 training zipped files and 2 validation zipped files into data folder and then run this script
"""
import cv2
import numpy as np
import os
import zipfile

# Running a loop through all the zipped training file to extract all video and then extract 100 frames from each.
for i in range(74, 76):
    if i < 10:
        zipfilename = 'training80_0' + str(i) + '.zip'
    else:
        zipfilename = 'training80_' + str(i) + '.zip'
    # Accessing the zipfile i
    archive = zipfile.ZipFile('chalearn/' + zipfilename, 'r')
    zipfilename = zipfilename.split('.zip')[0]

    # Extracting all videos in it and saving it all to the new folder with same name as zipped one
    archive.extractall('unzippedData/' + zipfilename)

    # Running a loop over all the videos in the zipped file and extracting 100 frames from each
    for file_name in archive.namelist():
        cap = cv2.VideoCapture('unzippedData/' + zipfilename + '/' + file_name)

        file_name = (file_name.split('.mp4'))[0]
        # Creating folder to save all the 100 frames from the video
        try:
            if not os.path.exists('ImageData/trainingData/' + file_name):
                os.makedirs('ImageData/trainingData/' + file_name)
        except OSError:
            print('Error: Creating directory of data')

        # Setting the frame limit to 100
        cap.set(cv2.CAP_PROP_FRAME_COUNT, 101)
        length = 101
        count = 0
        # Running a loop to each frame and saving it in the created folder
        while cap.isOpened():
            count += 1
            if length == count:
                break
            ret, frame = cap.read()
            if frame is None:
                continue

            # Resizing it to 256*256 to save the disk space and fit into the model
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
            # Saves image of the current frame in jpg file
            name = 'ImageData/trainingData/' + str(file_name) + '/frame' + str(count) + '.jpg'
            cv2.imwrite(name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print the file which is done
        print(zipfilename, ':', file_name)
# #
# for i in range(1, 26):
#     if i < 10:
#         zipfilename = 'validation80_0' + str(i) + '.zip'
#     else:
#         zipfilename = 'validation80_' + str(i) + '.zip'
#     # Accessing the zipfile i
#     archive = zipfile.ZipFile('data/' + zipfilename, 'r')
#     zipfilename = zipfilename.split('.zip')[0]
#
#     # Extracting all videos in it and saving it all to the new folder with same name as zipped one
#     archive.extractall('unzippedData/' + zipfilename)
#
#     # Running a loop over all the videos in the zipped file and extracting 100 frames from each
#     for file_name in archive.namelist():
#         cap = cv2.VideoCapture('unzippedData/' + zipfilename + '/' + file_name)
#
#         file_name = (file_name.split('.mp4'))[0]
#         # Creating folder to save all the 100 frames from the video
#         try:
#             if not os.path.exists('ImageData/validationData/' + file_name):
#                 os.makedirs('ImageData/validationData/' + file_name)
#         except OSError:
#             print('Error: Creating directory of data')
#
#         # Setting the frame limit to 100
#         cap.set(cv2.CAP_PROP_FRAME_COUNT, 101)
#         length = 101
#         count = 0
#         # Running a loop to each frame and saving it in the created folder
#         while cap.isOpened():
#             count += 1
#             if length == count:
#                 break
#             ret, frame = cap.read()
#             if frame is None:
#                 continue
#
#             # Resizing it to 256*256 to save the disk space and fit into the model
#             frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
#             # Saves image of the current frame in jpg file
#             name = 'ImageData/validationData/' + str(file_name) + '/frame' + str(count) + '.jpg'
#             cv2.imwrite(name, frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         # Print the file which is done
#         print(zipfilename, ':', file_name)
