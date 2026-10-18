import pytesseract
import cv2
from pprint import pprint
import math
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from os import listdir, makedirs

#-----------------------------
# IMAGE ENHANCEMENT
#-----------------------------
def image_enhance(FamID, input_dir, output_dir, upscale_factor = 10):
    '''
    Intakes raw pedigree image file and upscales for downstream analysis

    PARAMETERS:
    -----------
    FamID (string): ID denoting the particular pedigree for intake (should match PNG file name exactly)
    input_dir (string): file path where input pedigree image is found
    output_dir (string): file path to store upscaled images in
    upscale_factor (int): the degree to which the raw pedigree image should be upscaled (default = 10)

    RETURN:
    -------
    ourput_path (string): complete file path from wd to upscaled image
    '''
    input_path = f'{input_dir}/{FamID}.png'
    output_path = f'{output_dir}/{FamID}_upscaled{upscale_factor}x.png'

    img = Image.open(input_path)

    new_size = (img.width * upscale_factor, img.height * upscale_factor)
    upscaled_img = img.resize(new_size, Image.LANCZOS)

    makedirs(output_dir, exist_ok=True) 
    upscaled_img.save(output_path, quality= 95)

    return output_path

def linDist(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2

    return((x2-x1)**2+(y2-y1)**2)**0.5


def closestLabel(marker_coords, label_dict):
    closest_distance = math.inf
    closestLabel = None
    for label in label_dict.keys():
        label_x, label_y, label_w, label_h = label_dict[label]['label_coords']
        label_coords = ((label_x + label_w/2), (label_y + label_h/2))
        distance = linDist(marker_coords, label_coords)
        if distance < closest_distance:
            closest_distance = distance
            closestLabel = label
    return closestLabel



def merge_duplicate_lines(norm_cat_lines, angle_threshold= 10, distance_threshold= 50):
    
    merged_horizontal_lines = []
    for horz_line in norm_cat_lines['horizontal']:
        horz_duplicate = False
        a,hy1,b,hy2 = horz_line

        for i in range(len(merged_horizontal_lines)):
            c,mhy1,d,mhy2 = merged_horizontal_lines[i]
            
            #sufficiently close verically
            if abs(mhy1 - hy1) < distance_threshold:
                #forward subline
                if a <= c and d <= b:
                    merged_horizontal_lines[i] = (a,hy1,b,hy1)
                    horz_duplicate = True
                    break
                #overlap right
                elif a <= c and c <= b and b < d:
                    merged_horizontal_lines[i] = (a,hy1,d,hy1)
                    horz_duplicate = True
                    break
                #reverse subline
                elif c < a and b < d:
                    horz_duplicate = True
                    break
                #overlap left
                elif c < a and a <= d and d < b:
                    merged_horizontal_lines[i] = (c,hy1,b,hy2)
                    horz_duplicate = True
                    break

        if not horz_duplicate:
            merged_horizontal_lines.append(horz_line)

    
    merged_vertical_lines = []
    for vert_line in norm_cat_lines['vertical']:
        vert_duplicate = False
        vx1,a,vx2,b = vert_line

        for j in range(len(merged_vertical_lines)):
            mvx1,c,mvx2,d = merged_vertical_lines[j]
            
            #sufficiently close horizontally
            if abs(mvx1 - vx1) < distance_threshold:
                #forward subline
                if a >= c and d >= b:
                    merged_vertical_lines[j] = (vx1,a,vx1,b)
                    vert_duplicate = True
                #overlap top
                elif a >= c and c >= b and b > d:
                    merged_vertical_lines[j] = (vx1,a,vx1,d)
                    vert_duplicate = True
                #reverse subline
                elif c > a and b > d:
                    vert_duplicate = True
                #overlap bottom
                elif c >= a and a >= d and d > b:
                    merged_vertical_lines[j] = (vx1,c,vx1,b)
                    vert_duplicate = True

        if not vert_duplicate:
            merged_vertical_lines.append(vert_line)

    directional_merged_lines = {
        'vertical': merged_vertical_lines,
        'horizontal': merged_horizontal_lines
    }
    
    return directional_merged_lines



def categorize_normalize_lines(lines, img_width, img_height, tilt_threshold = 200):
    categorized_normalized_lines = {
    'vertical': [],
    'horizontal': []
    }

    for line in lines:
        x1,y1,x2,y2 = line[0]
        #deleting exessively long and short duplicates/artifacts
        if (
            (abs(x2-x1) > 0.8*img_width or abs(y2-y1) > 0.8*img_height) or
            (abs(x2-x1) < 0.01*img_width and abs(y2-y1) < 0.01*img_height)
        ):
            continue
        if abs(x1 - x2) < tilt_threshold:
            if y1 < y2:
                z = y1
                y1 = y2
                y2 = z
            categorized_normalized_lines['vertical'].append((x1, y1, x2, y2))
        elif abs(y1 - y2) < tilt_threshold:
            if x1 > x2:
                z = x1
                x1 = x2
                x2 = z                
            categorized_normalized_lines['horizontal'].append((x1, y1, x2, y2))
        else:
            print(f'This line does not fit categorization: ({line[0]})')

    return categorized_normalized_lines



def trackRelation(normalized_categorized_lines, IndvDataDict):
    connection_lines = []

    for IndvID in IndvDataDict.keys():
        IndvDataDict[IndvID]['PaternalID'] = 0
        IndvDataDict[IndvID]['MaternalID'] = 0
        inheritence_known = False
        #finding starting coordinate (top anchor)
        x_left, y_top, w, h = IndvDataDict[IndvID]['node_coords']
        x_mid = int(x_left + w/2)
        top_center_coord = (x_mid, y_top)
        IndvDataDict[IndvID]['lat_coords'] = [(int(x_left+w), int(y_top+(h/2))), (x_left, int(y_top+(h/2)))]
        distance_threshold = h/2
        #finding if there is veritcal line close to top anchor (i.e. if inheritence is known)
        for vert_line in normalized_categorized_lines['vertical']:
            Vx1,Vy1,Vx2,Vy2 = vert_line
            if linDist(top_center_coord, (Vx1,Vy1)) < distance_threshold:
                start_coord = (Vx1,Vy1)
                current_coord = (Vx2, Vy2)
                inheritence_known = True

                break

        #checking if parental relationship was found
        #tracing back to parents if known
        if inheritence_known:
            Cx,Cy = current_coord
            #find initial horizontal
            for horz_line in normalized_categorized_lines['horizontal']:
                Hx1,Hy1,Hx2,Hy2 = horz_line
                #checking if horizontal line is at same elevation as current y and current x is between enpoints
                if abs(Cy - Hy1) < distance_threshold and (Hx1-distance_threshold < Cx and Cx < Hx2+distance_threshold):
                    endpoints = horz_line
                    break
            
            #find secondary vertical (if it exists)
            secondary_exists = False
            Cx1,Cy1,Cx2,Cy2 = endpoints
            for vert_line in normalized_categorized_lines['vertical']:
                Vx1,Vy1,Vx2,Vy2 = vert_line
                #checks that the two lines intersect in a perpendicular way
                #adding distance threshold to Vy2 inorder to trncate the veritcal line from the top
                #this avoid seeing downward point sibling veritcal lines and progressing along that instead of parental lines
                if (Vy1+distance_threshold > Cy1 and Cy1 > Vy2+distance_threshold) and (Cx1-distance_threshold < Vx1 and Vx1 < Cx2+distance_threshold):
                    current_coord = (Vx2, Vy2)
                    secondary_exists = True
                    break
            
            #check if secondary vertical was found
            if secondary_exists:
                Cx,Cy = current_coord
                #find secondary horizontal
                for horz_line in normalized_categorized_lines['horizontal']:
                    Hx1,Hy1,Hx2,Hy2 = horz_line
                    if abs(Cy - Hy1) < distance_threshold and (Hx1-distance_threshold < Cx and Cx < Hx2+distance_threshold):
                        endpoints = horz_line
                        break
            
        
            fx1,fy1,fx2,fy2 = endpoints
            sx, sy = start_coord
            connection_lines = connection_lines + [(sx,sy,fx1,fy1), (sx,sy,fx2,fy2)]

            #determining indivudals associated with endpoints
            
            for ParentID in IndvDataDict.keys():
                #parent anchor points
                ParentCoords = IndvDataDict[ParentID]['node_coords']
                px_left, py_top, pw, ph = ParentCoords
                p_left_anchor = (px_left, py_top + ph/2)
                p_right_anchor = (px_left + pw, py_top + ph/2)
                if linDist(p_right_anchor, (fx1,fy1)) < distance_threshold:
                    IndvDataDict[IndvID]['PaternalID'] = ParentID
                    continue
                if linDist(p_left_anchor, (fx2,fy2)) < distance_threshold:
                    IndvDataDict[IndvID]['MaternalID'] = ParentID
                    continue

    return IndvDataDict, connection_lines

def pedigree_processing(FamID):
    #----------------------------------------
    # ENHANCE IMAGE
    #----------------------------------------
    upscale_factor = 10
    raw_image_dir = f'data/Pedigree_Images/Raw_Images'
    upscaled_image_dir = f'data/Pedigree_Images/Upscaled_Images'
    upscaled_image_path = image_enhance(FamID = FamID,
                                    input_dir= raw_image_dir,
                                    output_dir= upscaled_image_dir,
                                    upscale_factor= upscale_factor)

    #----------------------------------------
    # INDIVIDUAL ID DETECTION
    #----------------------------------------
    img = cv2.imread(upscaled_image_path)
    img_height, img_width, _ = img.shape
    img_area = img_height * img_width
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    redacted_img = np.copy(gray_img)
    TextData = pytesseract.image_to_data(gray_img, output_type= pytesseract.Output.DICT)
    IndvIDsDict = {}
    
    n_boxes = len(TextData['text'])

    for i in range(n_boxes):
        if len(TextData['text'][i]) > 0 and TextData['text'][i][0] != ' ':
            IndvID = TextData['text'][i]
            (x,y,w,h) = (TextData['left'][i], TextData['top'][i], TextData['width'][i], TextData['height'][i])
            display_coords = (x,y,w,h)
            IndvIDsDict[IndvID] = {}
            IndvIDsDict[IndvID]['label_coords'] = display_coords
            #draw a white rectangle over the ID number (with a little extra size for buffer)
            redacted_img = cv2.rectangle(redacted_img, (x-10,y-10), (x+w+20, y+h+20), (255,255,255), -1)


    #----------------------------------------
    # PHENOTYPE AND SEX DETECTION
    #----------------------------------------
    nodeless_img = np.copy(redacted_img)

    _, threshold_light = cv2.threshold(redacted_img, 250, 255, cv2.THRESH_BINARY)
    _, threshold_dark = cv2.threshold(redacted_img, 5, 255, cv2.THRESH_BINARY_INV)

    light_contours, _ = cv2.findContours(threshold_light, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dark_contours, _ = cv2.findContours(threshold_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    Phenotypes = ['+', '-']
    for phenotype in Phenotypes:
        contours = light_contours if phenotype == '-' else dark_contours
        for i, contour in enumerate(contours):
            #scaling epsilon used in polyaprox based on perimeter of the contour
            epsilon =0.02 * cv2.arcLength(contour, closed= True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            x,y,w,h = cv2.boundingRect(approx)
            bounding_area = w*h
            if bounding_area < 0.25*img_area and bounding_area > 0.0001*img_area and abs(w-h) < (min(w,h) * 0.25):

                nodeless_img = cv2.rectangle(nodeless_img, (x-35, y-40), (x+w+50, y+h+60), (255,255,255), -1)

                center_coords = ((x + w/2), y + h/2)
                label = closestLabel(marker_coords= center_coords, label_dict= IndvIDsDict)
                IndvIDsDict[label]['node_coords'] = (x,y,w,h)
                IndvIDsDict[label]['Phenotype'] = 1 if phenotype == '-' else 2
                xmid = int(x - 2*w)
                ybelow = int(y + 5*h/2)

                display_coords = (xmid, ybelow)


                if len(approx) == 4:
                    IndvIDsDict[label]['Sex'] = 1
                else:
                    IndvIDsDict[label]['Sex'] = 2


    #----------------------------------------
    # RELATION DETECTION
    #----------------------------------------
    edges = cv2.Canny(nodeless_img, 0, 50, apertureSize= 3)
    raw_lines = cv2.HoughLinesP(edges,
                            lines= np.array([]),
                            rho=1, 
                            theta= np.pi/180,
                            threshold= 50,
                            minLineLength= 5,
                            maxLineGap= img_width*0.1)

    line_img = np.copy(redacted_img)*0
    cat_norm_lines = categorize_normalize_lines(raw_lines, img_width, img_height)

    lines = merge_duplicate_lines(cat_norm_lines)
    for direction in lines.keys():
        for line in lines[direction]:
            x1, y1, x2, y2 = line
            line_img = cv2.line(line_img, (x1,y1), (x2,y2), (255,255,255), 5)
    
    IndvIDsDict, connection_lines = trackRelation(lines, IndvIDsDict)

    for line in connection_lines:
        x1, y1, x2, y2 = line
        line_img = cv2.line(line_img, (x1,y1), (x2,y2), (255,255,255), 5)
    for individual in IndvIDsDict.keys():
        for coord in IndvIDsDict[individual]['lat_coords']:
            line_img = cv2.circle(line_img, coord, radius=10, color=(255,255,255), thickness=-1)
    
    #----------------------------------------
    # Pedfile Generation
    #----------------------------------------
    makedirs('data/PedFiles/Automatic_Pedigrees', exist_ok= True)
    PedFile = f'data/PedFiles/Automatic_Pedigrees/{FamID}auto.ped'
    PedFileDataFields = ['PaternalID', 'MaternalID', 'Sex', 'Phenotype']
    with open(PedFile, 'w') as pf:
        for IndvID in IndvIDsDict.keys():
            pf.write(f'{FamID} {IndvID}')
            for field in PedFileDataFields:
                pf.write(f' {IndvIDsDict[IndvID][field]}')
            pf.write('\n')

    
    return redacted_img, nodeless_img, line_img


#Function Testing and Image Generation for Plots
if __name__ == '__main__':
    FamilyID = input('Eneter A Family ID for testing: ')
    print(f' Processing {FamilyID}')
    redacted_img, nodeless_img, line_img = pedigree_processing(FamilyID)
    

    cv2.imshow(f'{FamilyID} Redacted', redacted_img)
    cv2.imshow(f'{FamilyID} Nodeless', nodeless_img)
    cv2.imshow(f'{FamilyID} Relational Lines', line_img)

    k = cv2.waitKey(0)
    if k == ord('s'):
        image_path = 'data/CVProcessingImages'
        makedirs(image_path, exist_ok=True)
        cv2.imwrite(f'{image_path}/{FamilyID}redacted.png', redacted_img)
        cv2.imwrite(f'{image_path}/{FamilyID}nodesless.png', nodeless_img)
        cv2.imwrite(f'{image_path}/{FamilyID}lines.png', line_img)

    cv2.destroyAllWindows