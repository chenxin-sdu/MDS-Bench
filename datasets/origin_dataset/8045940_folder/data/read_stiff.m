function [cube, wavelengths, previewImage, metadata] = read_stiff(tiffFilename)
%READ_STIFF Load a hyperspectral image
    tiffObj = Tiff(tiffFilename);

    wavelengths = 0;
    metadata = 0;

    % MATLAB does not seem have a well-documented way to add new tags
    % to libtiff. Let's just root around in the file hoping to come across
    % our expected tags.
    info = imfinfo(tiffFilename);
    for i = 1:length(info)
        unknownTags = info(i).UnknownTags;
        for j = 1:length(unknownTags)
            if unknownTags(j).ID == 65000
                wavelengths = unknownTags(j).Value;
            elseif unknownTags(j).ID == 65111
                metadata = unknownTags(j).Value;
            end
        end
    end

    % The first page may contain an RGB image that should not
    % be part of the spectral image cube.
    if (getTag(tiffObj, 'Photometric') == Tiff.Photometric.RGB)
        previewImage = read(tiffObj);
        nextDirectory(tiffObj);
    end

    % Let's load the cube
    cube = [];
    while true
        cube = cat(3, cube, read(tiffObj));
        if lastDirectory(tiffObj)
            break
        end
        nextDirectory(tiffObj);
    end
end

