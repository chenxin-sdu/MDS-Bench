function [labels, masks, previewImage] = read_mtiff(tiffFilename)
%READ_MTIFF Load binary masks for a spectral image.
    tiffObj = Tiff(tiffFilename);

    masks = {};
    labels = {};

    % MATLAB does not seem have a well-documented way to add new tags
    % to libtiff. Let's just root around in the file hoping to come across
    % our expected tags.
    info = imfinfo(tiffFilename);
    for i = 1:length(info)
        unknownTags = info(i).UnknownTags;
        for j = 1:length(unknownTags)
            if unknownTags(j).ID == 65001
                labels = cat(1, labels, unknownTags(j).Value);
            end
        end
    end

    if (getTag(tiffObj, 'Photometric') == Tiff.Photometric.RGB)
        previewImage = read(tiffObj);
        nextDirectory(tiffObj);
    end

    % Load the masks
    while true
        masks = cat(1, masks, {read(tiffObj)});
        if lastDirectory(tiffObj)
            break
        end
        nextDirectory(tiffObj)
    end
end

