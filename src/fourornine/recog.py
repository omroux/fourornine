from shapely.geometry import Point, Polygon
import cv2

# the program may detect multiple rectangles, and the function should identify if two rectangles are virtually the same.


def same_rect(points1, points2):
    '''points1, points2 = sorted(points1),sorted(points2)
    for p1,p2 in zip(points1,points2):
        if (p2[1]-p1[1])**2+(p2[0]-p1[0])**2 > 1000:
            return False
    print(points1,points2)
    return True'''
    for p1 in points1:
        flag = True
        for p2 in points2:
            # 2000 - proximaty treshold.
            if (p2[1]-p1[1])**2+(p2[0]-p1[0])**2 < 2000:
                flag = False
        if flag:
            return False
    return True
# checks if one rectangle is completely inside of another.


def rect_inside(inner_rect, outer_rect):
    '''rinx,routx = sorted(rin),sorted(rout)
    riny,routy = sorted(rin, key=lambda p:(p[1],p[0])),sorted(rout, key=lambda p:(p[1],p[0]))
    return rinx[0][0] > routx[1][0] and rinx[3][0] < routx[2][0] and riny[0][1] > routy[1][1] and riny[3][1] < routy[2][1]'''
    poly = Polygon(outer_rect)
    for p in inner_rect:
        point = Point(p[0], p[1])
        if not poly.contains(point):
            return False
    return True
# gets the maximal depth in a graph. the graph should not contain any cycles.


def get_depth(graph, i):
    if len(graph[i]) == 0:
        return 0
    return max(get_depth(graph, j) for j in graph[i]) + 1
# returns if the pixel is a part of the digit or not.


def is_black(pixel):
    # return (pixel[0] == 0 or pixel[2]/pixel[0] >= 1.2) and (pixel[1] == 0 or pixel[2]/pixel[1] >= 1.2)
    return sum(pixel) < 350
# identify the digit given left middle and right parts of the digit, and low, low-middle high-middle and high parts.


def identify_digit(x1, x2, x3, y1, y2, y3, y4, image):
    lu, ld, u, m, d, ru, rd = True, True, True, True, True, True, True
    for i in range(-10, 10):
        flag1, flag2 = False, False
        for x in range(x1, x2):
            if is_black(image[y2+i][x]):
                flag1 = True
            if is_black(image[y3+i][x]):
                flag2 = True
        lu = lu and flag1
        ld = ld and flag2
        flag1, flag2 = False, False
        for x in range(x2, x3):
            if is_black(image[y2+i][x]):
                flag1 = True
            if is_black(image[y3+i][x]):
                flag2 = True
        ru = ru and flag1
        rd = rd and flag2
        flag1, flag2 = False, False
        for y in range(y1, y2):
            if is_black(image[y][x2+i]):
                flag1 = True
        u = u and flag1
        for y in range(y2, y3):
            if is_black(image[y][x2+i]):
                flag2 = True
        m = m and flag2
        flag1 = False
        for y in range(y3, y4):
            if is_black(image[y][x2+i]):
                flag1 = True
        d = d and flag1
    digits = [[True, True, True, False, True, True, True], [False, False, False, False, False, True, True], [False, True, True, True, True, True, False], [False, False, True, True, True, True, True], [True, False, False, True, False, True, True], [True, False, True,
                                                                                                                                                                                                                                                        True, True, False, True], [True, True, True, True, True, False, True], [False, False, True, False, False, True, True], [True, True, True, True, True, True, True], [True, False, True, True, True, True, True], [False, False, False, False, False, False, False]]
    for i, arr in enumerate(digits):
        if lu == arr[0] and ld == arr[1] and u == arr[2] and m == arr[3] and d == arr[4] and ru == arr[5] and rd == arr[6]:
            return i
    # print('d:',lu,ld,u,m,d,ru,rd)
    return -1
# given the rectangle that contains the 6 digits, identify all the digits inside of it.
# it returns the lines that where used for the seperation (for debugging) and the digits identified.


def seperate_digits(rect):
    rect = sorted(rect)
    y1 = (max(rect[0][1], rect[1][1])+max(rect[2][1], rect[3][1]))//2
    y2 = (min(rect[0][1], rect[1][1])+min(rect[2][1], rect[3][1]))//2
    x1, x2 = (rect[0][0] + rect[1][0])//2, (rect[2][0]+rect[3][0])//2
    dx, dy = x2-x1, y1 - y2
    lines = []
    lines.append(((int(x1 + dx*0.19), y1), (int(x1 + dx*0.19), y2)))
    lines.append(((int(x1 + dx*0.29), y1), (int(x1 + dx*0.29), y2)))
    lines.append(((int(x1 + dx*0.4), y1), (int(x1 + dx*0.4), y2)))
    lines.append(((int(x1 + dx*0.5), y1), (int(x1 + dx*0.5), y2)))
    lines.append(((int(x1 + dx*0.6), y1), (int(x1 + dx*0.6), y2)))
    lines.append(((int(x1 + dx*0.72), y1), (int(x1 + dx*0.72), y2)))
    lines.append(((int(x1 + dx*0.82), y1), (int(x1 + dx*0.82), y2)))
    lines.append(((x1, int(y2+dy/10.0)), (x2, int(y2+dy/10.0))))
    lines.append(((x1, int(y2+dy/3.0)), (x2, int(y2+dy/3.0))))
    lines.append(((x1, int(y2+2*dy/3.0)), (x2, int(y2+2*dy/3.0))))
    lines.append(((x1, int(y2+9*dy/10.0)), (x2, int(y2+9*dy/10.0))))
    lines.append(((int(x1 + dx*0.24), y1), (int(x1 + dx*0.24), y2)))
    lines.append(((int(x1 + dx*0.34), y1), (int(x1 + dx*0.34), y2)))
    lines.append(((int(x1 + dx*0.45), y1), (int(x1 + dx*0.45), y2)))
    lines.append(((int(x1 + dx*0.55), y1), (int(x1 + dx*0.55), y2)))
    lines.append(((int(x1 + dx*0.67), y1), (int(x1 + dx*0.67), y2)))
    lines.append(((int(x1 + dx*0.77), y1), (int(x1 + dx*0.77), y2)))
    d1 = identify_digit(int(x1 + dx*0.19), int(x1 + dx*0.24), int(x1 + dx*0.29),
                        int(y2+dy/10.0), int(y2+dy/3.0), int(y2+2*dy/3.0), int(y2+9*dy/10.0), image)
    d2 = identify_digit(int(x1 + dx*0.29), int(x1 + dx*0.34), int(x1 + dx*0.4),
                        int(y2+dy/10.0), int(y2+dy/3.0), int(y2+2*dy/3.0), int(y2+9*dy/10.0), image)
    d3 = identify_digit(int(x1 + dx*0.4), int(x1 + dx*0.45), int(x1 + dx*0.5), int(
        y2+dy/10.0), int(y2+dy/3.0), int(y2+2*dy/3.0), int(y2+9*dy/10.0), image)
    d4 = identify_digit(int(x1 + dx*0.5), int(x1 + dx*0.55), int(x1 + dx*0.6), int(
        y2+dy/10.0), int(y2+dy/3.0), int(y2+2*dy/3.0), int(y2+9*dy/10.0), image)
    d5 = identify_digit(int(x1 + dx*0.6), int(x1 + dx*0.67), int(x1 + dx*0.72),
                        int(y2+dy/10.0), int(y2+dy/3.0), int(y2+2*dy/3.0), int(y2+9*dy/10.0), image)
    d6 = identify_digit(int(x1 + dx*0.72), int(x1 + dx*0.77), int(x1 + dx*0.82),
                        int(y2+dy/10.0), int(y2+dy/3.0), int(y2+2*dy/3.0), int(y2+9*dy/10.0), image)
    # print(d1%10,d2%10,d3%10,d4%10,d5%10,d6%10)
    return lines, [d1, d2, d3, d4, d5, d6]


# code from chatgpt, don't tell anyone ;)
# Load the image
image_path = 'image11.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge map
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours based on their area and shape
min_area = 50000  # Minimum contour area to consider
# Minimum number of sides (approximated) to consider as a potential square
min_sides = 4

potential_squares = []
for contour in contours:
    area = cv2.contourArea(contour)
    approx = cv2.approxPolyDP(
        contour, 0.04 * cv2.arcLength(contour, True), True)
    sides = len(approx)
    if area > min_area and sides == min_sides:
        potential_squares.append(approx)


# Get the corner points for each potential square
corner_points = []
for square in potential_squares:
    corners = []
    for point in square:
        x, y = point[0]
        corners.append((x, y))
    corner_points.append(corners)
is_real_corners = [True] * len(corner_points)
for i, points in enumerate(corner_points):
    if not is_real_corners[i]:
        continue
    for j, points2 in enumerate(corner_points):
        if j <= i:
            continue
        if same_rect(points, points2):
            is_real_corners[j] = False
real_corners = []
for i, points in enumerate(corner_points):
    if is_real_corners[i]:
        real_corners.append(points)
# finds a DAG that represents which rectangle is inside of which.
rect_graph = [[] for i in range(len(real_corners))]
for i, r1 in enumerate(real_corners):
    for j, r2 in enumerate(real_corners):
        if i >= j:
            continue
        if rect_inside(r1, r2):
            rect_graph[j].append(i)
        elif rect_inside(r2, r1):
            rect_graph[i].append(j)

# Print the corner points of each polygon
'''for i, corners in enumerate(real_corners):
    d = get_depth(rect_graph,i)
    if d==0:
        slines = seperate_digits(corners)
        for l in slines:
            cv2.line(image,l[0],l[1],(0,0,0),1)
        for i in range(len(slines)-1):
            #roi = image[slines[i][1][1]:slines[i][0][1], slines[i][0][0]:slines[i+1][0][0]]
            ''roi = image[0:1000,0:1000]
            gray_roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi,50,150,apertureSize=3)
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
            if type(lines) != NoneType:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(image, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 2)''
    for point in corners:
        cv2.circle(image,(point[0],point[1]),3,(255*int(d==0),255*int(d==1),255*int(d>=2)),-1)'''
pages = []
for i, corners in enumerate(real_corners):
    d = get_depth(rect_graph, i)
    for point in corners:
        cv2.circle(image, (point[0], point[1]), 3, (255 *
                   int(d == 0), 255*int(d == 1), 255*int(d >= 2)), -1)
    # only rectangles that contain 5 rectangles inside of them, and are of depth 1, are the pages in the image.
    if d == 1:
        if len(rect_graph[i]) != 5:
            print('whaaaaat', len(rect_graph[i]))
            continue
        pages.append([])
        for j in sorted(rect_graph[i], key=lambda k: (sorted(real_corners[k], key=lambda p: p[1])[0][1])):
            slines, digits = seperate_digits(real_corners[j])
            if -1 in digits:
                pages[-1].append(['e', 'r', 'r', 'o', 'r', '!'])
                pages[-1].append(digits)
            else:
                pages[-1].append(digits)
            for l in slines:
                cv2.line(image, l[0], l[1], (0, 0, 0), 1)
# print the results.
for i, p in enumerate(pages):
    print('Page', i, 'times:')
    for digits in p:
        for d in digits:
            if (d == 10):
                print('-', end="")
            else:
                print(d, end="")
        print()

# Display the image with the potential squares highlighted
cv2.imwrite('res11.jpg', image)
cv2.imshow("Potential Squares", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
