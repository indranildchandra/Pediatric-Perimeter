%% Fast Radial Symmetry Transform function
% This algorithm segments and returns all the possible locations of the 
% center of the Pupil in the image passed to it as an argument
%%

function [filteredImage] = fastRadialSymmetryTransform(Image, pupilRadii, param, stdDev)
    originalImage = double(Image);
    [gy, gx] = gradient(originalImage);
    maxRadius = ceil(max(pupilRadii(:)));
    offset = [maxRadius maxRadius];
    filteredImage = zeros(size(originalImage) + 2*offset);
    S = zeros([numel(pupilRadii), size(filteredImage, 1), size(filteredImage, 2)]);
    
    radiusIndex = 1;
    for n = pupilRadii 
        O = zeros(size(filteredImage));
        M = zeros(size(filteredImage));
        for i = 1:size(originalImage, 1)
            for j=1:size(originalImage, 2)
                currentPoint = [i j];
                grad = [gx(i,j) gy(i,j)];
                gradientNorm = sqrt( grad * grad' ) ;
                if (gradientNorm > 0)
                    gradientPattern = round((grad./gradientNorm) * n);       
                    diff = currentPoint - gradientPattern;
                    diff = diff + offset;
                    O(diff(1), diff(2)) = O(diff(1), diff(2)) - 1;
                    M(diff(1), diff(2)) = M(diff(1), diff(2)) - gradientNorm;
                end 
            end
        end
        
        M = abs(M);
        O = abs(O);
        M = M ./ max(M(:));
        O = O ./ max(O(:));
        S_n = (O.^param) .* M;
        gaussian = fspecial('gaussian', [ceil(n/2) ceil(n/2)], n*stdDev);
        S(radiusIndex, :, :) = imfilter(S_n, gaussian);
        radiusIndex = radiusIndex + 1;
    end
    
    filteredImage = squeeze(sum(S, 1));
    filteredImage = filteredImage(offset(1)+1:end-offset(2), offset(1)+1:end-offset(2));

end

