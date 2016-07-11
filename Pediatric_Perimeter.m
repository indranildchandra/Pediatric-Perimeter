%% Pediatric Perimeter Pupil Segmentation Algorithm
% Author: Indranil Chandra
% The algorithm was developed during MIT-LVPEI "Engineering the Eye
% Workshop" - 2016 at Srujana Innovation Center, Hyderabad
%%

close all;
clear all;
clc;

elapsedTime = zeros(1,11);
IPD = zeros(1,11);
Skew = zeros(1,11);
Angle = zeros(1,11);
baseIPD = 52.2251;
baseAngle = radtodeg(atan( 39.5 - 23.30 / 32 - 81.65 ));

% Select the minimum size above which nose will be detected
noseThreshold = 1; % Default = 16

for frameNum = 1:1:11
    imageName = strcat('testImage',num2str(frameNum),'.png');
    frame = imread(imageName);
%     figure, imshow(frame),
%     title('Original Image');
%     impixelinfo;

    tic;
    % Detect Face
    faceDetect = vision.CascadeObjectDetector;
    BB_face_all = step(faceDetect,frame);
    BB_face = zeros(1,4);
    [faceCount, cnt1] = size(BB_face_all);
    caption = 'Detected Face Location';

    if isempty(BB_face_all)
        BB_face(1,1) = floor((10/100)*size(frame,1));
        BB_face(1,2) = floor((10/100)*size(frame,2));
        BB_face(1,3) = size(frame,1) - floor((10/100)*size(frame,1));
        BB_face(1,4) = size(frame,2) - floor((10/100)*size(frame,2));
%         figure, imshow(frame),
%         title('No Face Detected');
%         impixelinfo;

        I = imcrop(frame,BB_face(1,:));
        grayImage = rgb2gray(I);
        caption = 'Face not Detected';
    else
        if faceCount > 1             %Discard faces detected in the background
%             figure, imshow(frame),
%             title('All Detected Faces');
%             impixelinfo;
%             for i=1:1:size(BB_face_all,1)
%                 rectangle('Position',BB_face_all(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','r');
%             end

            caption = 'Multiple Faces Detected: Most Prominent selected';
            
            Area = zeros(size(BB_face_all,1));
            for i = 1:size(BB_face_all,1)
                l = BB_face_all(i,3);
                b = BB_face_all(i,4);
                Area(i,1) = l*b;
            end
            maxArea = max(Area);
            for i = 1:1:size(BB_face_all,1)
                if (Area(i,1)==maxArea(1))
                    for k = 1:4
                        BB_face(1,k) = BB_face_all(i,k);
                    end
                end
            end

            BB_face(1,1) = BB_face(1,1) + floor((15/100)*BB_face(1,3));
            BB_face(1,2) = BB_face(1,2) + floor((15/100)*BB_face(1,4));
            BB_face(1,3) = BB_face(1,3) - floor((15/100)*BB_face(1,3));
            BB_face(1,4) = BB_face(1,4) - floor((15/100)*BB_face(1,4));
%             figure, imshow(frame),
%             title('Most Dominant Face');
%             impixelinfo;
%             rectangle('Position',BB_face(1,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
        else
            BB_face(1,1) = BB_face_all(1,1) + floor((15/100)*BB_face_all(1,3));
            BB_face(1,2) = BB_face_all(1,2) + floor((15/100)*BB_face_all(1,4));
            BB_face(1,3) = BB_face_all(1,3) - floor((15/100)*BB_face_all(1,3));
            BB_face(1,4) = BB_face_all(1,4) - floor((15/100)*BB_face_all(1,4));
%             figure, imshow(frame),
%             title('Detected Face');
%             impixelinfo;
%             rectangle('Position',BB_face(1,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
        end

        I = imcrop(frame,BB_face(1,:));
        grayImage = rgb2gray(I);
        
% -------------------------------------------------------------------------
        %To detect Nose
        noseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',noseThreshold);
        BB_nose1=step(noseDetect,grayImage);

            Area = zeros(size(BB_nose1,1));
            for i = 1:size(BB_nose1,1)
                l = BB_nose1(i,3);
                b = BB_nose1(i,4);
                Area(i,1) = l*b;
            end
            maxArea = max(Area);
            length = size(BB_nose1,1);
            j=1;
            for i = 1:length
                if (Area(i,1)==maxArea(1))
                    for k = 1:4
                        BB_nose(j,k) = BB_nose1(i,k);
                    end
                end
            end
        end

% -------------------------------------------------------------------------

    f = fastRadialSymmetryTransform(grayImage, 7, 2, 0.01);
%     figure, subplot(2,1,1);
%     imshow(grayImage, []);
%     subplot(2,1,2);
%     imshow(f,[]),
%     impixelinfo;

    level = graythresh(f);
    binaryMask= im2bw(f,level);
%     figure, imshow(binaryMask), title('Binary Mask');

    binaryMaskNew = binaryMask;
    CC = bwconncomp(binaryMaskNew);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [connComps,idx] = sort(numPixels,'descend');
    binaryMask(CC.PixelIdxList{idx(1)}) = 0;
    binaryMask(CC.PixelIdxList{idx(2)}) = 0;
    filteredImage = imsubtract(binaryMaskNew,binaryMask);
    [labels, num] = bwlabel(filteredImage);

%     figure, imshow(grayImage), title('Probable Pupil Locations'), hold on,
    for cnt = 1:1:num
        s = regionprops(labels, 'BoundingBox', 'Area', 'Centroid','MajorAxisLength','MinorAxisLength');
%         rectangle('position', s(cnt).BoundingBox,'EdgeColor','b','linewidth',2);
    end
%     impixelinfo;

    edgeImage = edge(grayImage, 'Canny');
%     figure, imshow(edgeImage), title('Canny Edge Image'), hold on;
%     impixelinfo;

    pupilCenters(1,1) = s(1).Centroid(1);
    pupilCenters(1,2) = s(1).Centroid(2);
    pupilCenters(2,1) = s(2).Centroid(1);
    pupilCenters(2,2) = s(2).Centroid(2);

    leftBound1 = floor(s(1).BoundingBox(1) - 10);
    rightBound1 = floor(s(1).BoundingBox(1) + s(1).BoundingBox(3) + 10);
    upBound1 = floor(s(1).BoundingBox(2) - 10);
    downBound1 = floor(s(1).BoundingBox(2) + s(1).BoundingBox(4) + 10);
    leftBound2 = floor(s(2).BoundingBox(1) - 10);
    rightBound2 = floor(s(2).BoundingBox(1) + s(2).BoundingBox(3) + 10);
    upBound2 = floor(s(2).BoundingBox(2) - 10);
    downBound2 = floor(s(2).BoundingBox(2) + s(2).BoundingBox(4) + 10);

    edgeMask = zeros(size(edgeImage));

    for i = leftBound1:1:rightBound1
        for j = upBound1:1:downBound1
            edgeMask(j,i) = 1;
        end
    end
    for i = leftBound2:1:rightBound2
        for j = upBound2:1:downBound2
            edgeMask(j,i) = 1;
        end
    end
%     figure, imshow(edgeMask), title('Edge Mask'), hold on;
%     impixelinfo;


    edgeImageMask = zeros(size(edgeImage));

    for i = 1:1:size(edgeImage,1)
        for j = 1:1:size(edgeImage,2)
            edgeImageMask(i,j) = edgeMask(i,j) .* edgeImage(i,j);
        end
    end
%     figure, imshow(edgeImageMask), title('Edge Image Mask'), hold on;
%     impixelinfo;

    mask = zeros(size(edgeImageMask));
    finalImage = zeros(size(edgeImage));
    for i = 1:1:size(edgeImageMask,1)
        for j = 1:1:size(edgeImageMask,2)
            if edgeImageMask(i,j) == 1
                mask(i,j) = 255;
                finalImage(i,j) = grayImage(i,j) + mask(i,j);
                if finalImage(i,j) > 255
                    finalImage(i,j) = 255;
                end
            end
        end
    end

    figure, subplot(1,2,1), imshow(grayImage), title(caption);
    if (~isempty(BB_face_all) && ~isempty(BB_nose))
        for i=1:1:size(BB_nose,1)
            rectangle('Position',BB_nose(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','r');
        end
    end

    
    subplot(1,2,2), imshow(finalImage), title('Pupil Locations'), hold on,
    for cnt = 1:1:num
        s = regionprops(labels, 'BoundingBox', 'Area', 'Centroid','MajorAxisLength','MinorAxisLength');
        rectangle('position', s(cnt).BoundingBox,'EdgeColor','b','linewidth',2);
    end
    impixelinfo;
%     saveas(gcf,'PediPeri_result','png')
    
    currentIPD = sqrt((pupilCenters(1,1) - pupilCenters(2,1))^2 + (pupilCenters(1,2) - pupilCenters(2,2))^2);
    IPD(1,frameNum) = currentIPD;
    currentSkew = radtodeg(acos(baseIPD/currentIPD));
    Skew(1,frameNum) = round(currentSkew);
    currentAngle = radtodeg(atan( pupilCenters(1,2) - pupilCenters(2,2) / pupilCenters(1,1) - pupilCenters(2,1) ));
    Angle(1,frameNum) = round(currentAngle - baseAngle);
    
    toc;
    elapsedTime(1,frameNum) = toc;
    pause(3);
end

Avearage_Time = mean(elapsedTime(2:1:11));
display('Average Time Taken to Segment each image (in seconds): ');
display(Avearage_Time);

