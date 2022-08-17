#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>
#include <stack>
#include <iterator>
#include <list>
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
// //#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb/stb_image_resize.h"
#include "../stb/stb_image_write.h"
using namespace std;

// namespace letters{
    /**
     * @brief structure holding data on letters and their location in the image
     * 0 ____________
     * 1 __XXXXXXXX__
     * 2 _____XX_____
     * 3 _____XX_____
     * 4 _____XX_____
     * 5 _____XX_____
     * 6 ____________
     *   0 1 2 3 4 5
     * minHight - the lowest pixels position of the letter in the image, i.e. 5
     * maxHight - the highest pixels position of the letter in the image, i.e. 1
     * minWidth - the left most pixels position of the letter in the image, i.e. 1
     * maxWidth - the right most pixels position of the letter in the image, i.e. 4
     * lineNum - the line in which the letter is positioned
     * letterNum - the index of the letter in the line
     * pixels - an array holdind the pixel values of the letter 
     */
    struct letter_data {
        int minHeight,
            maxHeight,
            minWidth,
            maxWidth,
            lineNum,
            letterNum;
        uint8_t* pixels;

        bool operator!=(const letter_data other) {
            bool res =  (minHeight == other.minHeight) &&
                        (maxHeight == other.maxHeight) &&
                        (minWidth  == other.minWidth)  &&
                        (maxWidth  == other.maxWidth);
            return res;
        }
    };

    int     *lines,
            *tempLines;
    const int LINE_RAD = 3,
            MIN_PIXEL = 1,
            MIN_PIXEL_LONG = 12,
            BLACK = 0,
            WHITE = 255,
            PIX_28 = 28,
            LIGHT_GREY = 100;
    uint8_t *visitedPixels,
            *line_data,
            *image_grey;
    stbi_uc *image, *tempImage;


    list<letter_data*> letters;
    list<letter_data*> letters2;

    const string    PATH_IN = "../resources/in/", //C:/Users/Itay/source/repos/ConsoleApplication2
                    PATH_OUT = "../resources/out/", //C:/Users/Itay/source/repos/ConsoleApplication2
                    IMG_NAME = "avg_func_gray.jpeg",
                    IMG_LINE = "line_",
                    IMG_IDX = "_index_",
                    IMG_IN = "../resources/in/avg_func.jpg", //C:/Users/Itay/source/repos/ConsoleApplication2
                    IMG_IN_DIGIT = "../resources/in/digits2.jpg", //C:/Users/Itay/source/repos/ConsoleApplication2
                    IMG_OUT = "../resources/out/avg_func.bmp", // C:/Users/Itay/source/repos/ConsoleApplication2
                    IMG_OUT2 = "../resources/out/avg_func2.bmp", // C:/Users/Itay/source/repos/ConsoleApplication2
                    IMG_BMP = ".bmp",
                    IMG_JPEG = ".jpeg",
                    IMG_PNG = ".png";

    // /**
    //  * @brief variabels for the loaded image
    //  * width, height - dimentions of the image
    //  * channels - the layers of the image
    //  * numLines - the number of lines found in the image
    //  */
    int     width = 0,
            height = 0,
            channels = 0,
            numLines = 0;

    // void ExtractLetter::convertTo28(); 

    /*
    * Reads an image and makes it negative, white -> black & black -> white.
    */
    void imgToArray() {
        //char* c = const_cast<char*>(IMG_IN.c_str());
        char* c = const_cast<char*>(IMG_IN_DIGIT.c_str());
        // stbi_set_flip_vertically_on_load(true);
        tempImage = stbi_load(c, &width , &height , &channels , 3);  /*PATH_IN + *IMG_NAME*/
        // image     = stbi_load(c, &width2, &height2, &channels2, 3);  /*PATH_IN + *IMG_NAME*/
        
        

        //image_gray = make_unique<int[]>(height * width);
        int k = 0, idx = 0;
        //cout << channels << " " << width << " " << height << "\n";
        for (uint8_t* p = tempImage; p != tempImage + ((width) * (height) * (channels)); p += channels) {

            *p = WHITE - *p;
            *(p + 1) = WHITE - *(p + 1);
            *(p + 2) = WHITE - *(p + 2);
            //idx++;

            //if (k < width * height && (idx % 3 == 0)) {
            //    //image_gray[i]  = (int)((*p * 0.3) + ((*(p + 1) * 0.59)) + (*(p + 2) * 0.11));
            //    image_gray[k] = 255 - (*p * 0.3) + (*(p + 1) * 0.59) + (*(p + 2) * 0.11);
            //    //if (image_gray[k] < 356) 
            //    cout << image_gray[k] << "\n";
            //    k++;
            //}
        }
        
        //cout << idx << "\n";
        //stbi_write_bmp(IMG_OUT, width, height, channels, image /* quality = 100 */);  /*PATH_OUT + *IMG_NAME*/
        //stbi_write_bmp("C:/Users/Itay/source/repos/ConsoleApplication2/resources/out/avg_func_out.bmp", width, height, 1, image_gray /* quality = 100 */);  /*PATH_OUT + *IMG_NAME*/
    }


    /*
    * Gets a whole image containing (at least one) lines of pseudo code and splits the into seperate lines and saves it to 'int * lines'.
    * Does it by calculating for each line a representitive value of all the pixels, after that and according to the values
    * we can know when a line starts and where it ends.
    */
    void seperateLines() {
        line_data = new uint8_t[height];
        image_grey = new uint8_t[height * width];
        int sum = 0, idx = 0, greyIdx = 0, widCounter = 0, heiCounter = 0, lineMin = 0, lineMax = 0, lineCounter = 0, gapCounter = 0;
        float div = width * 0.5;
        bool lineFlag = false;
        for (int i = 0; i < (height * width * channels); i += channels) {
            //for (int j = 0; j < width; j++) {
                //for(int k = 0; k < channels; k++)
            int tempVal = (tempImage[i + 0] * 0.30) +
                (tempImage[i + 1] * 0.59) +
                (tempImage[i + 2] * 0.11);
            sum += tempVal;
            if (greyIdx < height * width) {
                image_grey[greyIdx++] = tempVal;
            }
            widCounter++;
            //sum += image[i * height + j];
        //}
            if (widCounter == width && idx < height) {
                //cout << idx << " : " /* << widCounter << "\t"*/ << sum << "\t" << (int)(sum / (width * 0.1)) << "\n";
                line_data[idx++] = (int)(sum / (width * 0.1));
                sum = 0;
                widCounter = 0;
            }
        }

        image = new uint8_t(height * width);
        cout << "before switch\n";
        for (int i = 0; i < height; i++)
        {
                cout << i << " -> " << height - i - 1 << "\n";

            for (int j = 0; j < width; j++)
            {
                // cout << i * height + j << " -> " << height - i - 1 << " x " << j << "\n";
                image[i * width + j] = image_grey[(height - i - 1) * width + j];
            }
        }
        
        cout << "after switch\n";

        stbi_write_bmp("../resources/out/main.bmp", width, height, channels, image);
        char* c = const_cast<char*>(IMG_OUT2.c_str());
        // cout << "printing whole image\n";
        // stbi_write_bmp(c, width, height, 1, image_grey /* quality = 100 */);
        tempLines = new int[idx / (LINE_RAD * 2)];
        for (int i = 0; i < idx; i++) {
            if (line_data[i] > MIN_PIXEL && !lineFlag) {
                lineFlag = true;
                heiCounter++;
                lineMin = i;
            }
            else if (lineFlag && line_data[i] > MIN_PIXEL) {
                heiCounter++;
                gapCounter = 0;
            }
            else if (lineFlag && line_data[i] <= MIN_PIXEL && gapCounter < LINE_RAD) {
                heiCounter++;
                gapCounter++;
            }
            else if (lineFlag && line_data[i] <= MIN_PIXEL && heiCounter > LINE_RAD) {
                lineMax = i;
                tempLines[lineCounter] = (int)lineMin;
                tempLines[lineCounter + 1] = (int)lineMax;
                //cout << (int)tempLines[lineCounter] << " " << (int)tempLines[lineCounter+1] << "\n";
                lineCounter += 2;
                heiCounter = 0;
                lineFlag = false;
                lineMin = 0;
                lineMax = 0;
            }
            else /*if (lineFlag && line_data[i] <= MIN_PIXEL && heiCounter <= LINE_RAD)*/ {
                heiCounter = 0;
                lineFlag = false;
                lineMin = 0;
                lineMax = 0;
            }
        }
        lines = tempLines;
        //lines = new int[lineCounter];
        //for (int i = 0; i < lineCounter /* && i < sizeof(tempLines)*/; i++) { // lineCounter
        //    int t1 = tempLines[i];
        //    lines[i] = tempLines[i];
        //    int t2 = lines[i];
        //    //cout << "";
        //}
        numLines = lineCounter / 2;
        //delete[] tempLines;
    }

    /*
    * Adds all the closests unvisited neighboring pixels of a given pixel to a stack.
    * this function is used be the letter seperating algorithm.
    */
    void addNeighborsToStack(int i, int j, stack<int>* stack) {
        for (int k = i - 1; k < i + 2; k++) {
            for (int l = j - 1; l < j + 2; l++) {
                int t = visitedPixels[k * width + l];
                if (!((k == i) && (l == j)) && (k < height) && (l < width) && (k > 0) && (l > 0) && (visitedPixels[k * width + l] == 0)) {
                    stack->push(k * width + l);
                }
            }
        }
    }

    /*
    * Searches for relevant pixel in concern to a letter, in other words searches and saves all the pixels that are part of a letter.
    */
    void searchNeighbors(stack<int>* stack) {

        list<int> goodPixels;
        int heightMin = INT_MAX,
            heightMax = 0,
            widthMin = INT_MAX,
            widthMax = 0,
            newHeight = 0,
            newWidth = 0,
            letterIdx = 0;

        while (!stack->empty()) {
            int idx = stack->top();
            stack->pop();
            int i = idx / width,
                j = idx % width;
            if (i < heightMin) {
                heightMin = i;
            }
            else if (i > heightMax) {
                heightMax = i;
            }
            if (j < widthMin) {
                widthMin = j;
            }
            else if (j > widthMax) {
                widthMax = j;
            }
            if (idx < width * height) {
                visitedPixels[idx] = 1;
            }
            int t = image_grey[idx];
            if (idx < width * height && image_grey[idx] > LIGHT_GREY) {
                goodPixels.push_back(idx);
                // Adding neighbors to the stack //
                addNeighborsToStack(i, j, stack);
            }
        }

        newHeight = heightMax - heightMin;
        newWidth = widthMax - widthMin;
        if (newHeight > MIN_PIXEL && newWidth > MIN_PIXEL) {
            uint8_t* letter = new uint8_t[newHeight * newWidth];
            for (int k = 0; k < newHeight * newWidth; k++) {
                letter[k] = BLACK;
            }
            //stbi_write_bmp("C:/Users/Itay/source/repos/ConsoleApplication2/resources/out/avg_func_Black.bmp", newWidth, newHeight, 1, letter /* quality = 100 */);

            list<int>::iterator it;
            for (it = goodPixels.begin(); it != goodPixels.end(); ++it) {
                int i = (*it / width) - heightMin,
                    j = (*it % width) - widthMin;
                letter[i * newWidth + j] = image_grey[*it];
                letterIdx++;
            }

            //std::cout << letterIdx << "\n";
            letter_data* tempLetter = new letter_data;
            tempLetter->maxHeight = heightMax;
            tempLetter->minHeight = heightMin;
            tempLetter->maxWidth = widthMax;
            tempLetter->minWidth = widthMin;
            tempLetter->pixels = letter;
            letters.push_back(tempLetter);
            letters2.push_back(tempLetter);

            /*for (int i = 0; i < newHeight; i++) {
                for (int j = 0; j < newWidth; j++){
                    if(letter[i * newWidth + j] > LIGHT_GREY)
                        cout << "1";
                    else {
                        cout << "0";
                    }
                }
                cout << "\n";
            }*/

            //const std::string s = std::to_string(letterIdx);
            //const char* pchar = s.c_str();
            //char* address = "C:/Users/Itay/source/repos/ConsoleApplication2/resources/out/avg_func_" + *pchar;
            //address += ".bmp";
            //stbi_write_bmp("C:/Users/Itay/source/repos/ConsoleApplication2/resources/out/avg_func_.bmp", newWidth, newHeight, 1, letter /* quality = 100 */);
        }
    }

    /*
    *
    */
    void seperateLetters() {
        stack<int> stack;
        visitedPixels = new uint8_t[height * width];
        for (int i = 0; i < height * width; i++) {
            visitedPixels[i] = 0;
        }
        // cout << "1\n";
        int lineHeight = 0, lineNum = 0, letterNum = 0;
        for (int k = 0; k < numLines * 2; k += 2) {
            //lineHeight = lines[k + 1] - lines[k];
            //int q1 = lines[k], q2 = lines[k + 1];
            //cout <<  q1 << " - " << q2 << "\n";
            // cout << numLines << "\n";
            for (int j = 0; j < width; j++) {
                for (int i = lines[k]; i < lines[k + 1]; i++) {
                    if (visitedPixels[i * width + j] == 0 && image_grey[i * width + j] > LIGHT_GREY) {
                        //int temp[2] = { i, j };
                        stack.push(i * width + j);
                    searchNeighbors(&stack);
                        //delete[] temp;
                    }
                }
            }
                // cout << "2";
            letter_data* oldIt = new letter_data;
            bool flag = false;
            for (letter_data* it : letters) {
                // letters2.push_back(it);
                if (flag) {
                    if (abs(oldIt->minWidth - it->minWidth) < MIN_PIXEL_LONG || abs(it->maxWidth - oldIt->maxWidth) < MIN_PIXEL_LONG ||
                        (oldIt->minHeight > it->maxHeight && (oldIt->maxWidth + MIN_PIXEL_LONG < it->minWidth)) ||
                        (it->minHeight < oldIt->maxHeight && (it->minWidth + MIN_PIXEL_LONG < oldIt->maxWidth))) {
                        //(abs(it->maxWidth - oldIt->minWidth) < MIN_PIXEL && (abs(it->maxHeight - oldIt->minHeight) < 1 || abs(it->minHeight - oldIt->minHeight) < 1)) ||
                        //(abs(oldIt->maxWidth - it->minWidth) < MIN_PIXEL && (abs(it->maxHeight - oldIt->minHeight) < 1 || abs(it->minHeight - oldIt->minHeight) < 1))) {

                        int     minWidth = min(oldIt->minWidth, it->minWidth),
                                maxWidth = max(oldIt->maxWidth, it->maxWidth),
                                minHeight = min(oldIt->minHeight, it->minHeight),
                                maxHeight = max(oldIt->maxHeight, it->maxHeight),
                                newWidth = maxWidth - minWidth,
                                newHeight = maxHeight - minHeight;
                        uint8_t *newPixels = new uint8_t[(maxWidth - minWidth) * (maxHeight - minHeight)];
                        for (int i = 0; i < (maxWidth - minWidth) * (maxHeight - minHeight); i++) {
                            newPixels[i] = BLACK;
                        }
                        int itWidth = it->maxWidth - it->minWidth,
                            itHeight = it->maxHeight - it->minHeight,
                            itWidthOffset = abs(minWidth - it->minWidth),
                            itHeightOffset = abs(minHeight - it->minHeight);
                        for (int i = 0; i < itHeight; i++) {
                            for (int j = 0; j < itWidth; j++) {
                                //int q1 = i * itWidth + j,
                                //    q2 = i * itWidth + itHeightOffset + j + itWidthOffset,
                                //    v1 = it->pixels[i * itWidth + j];
                                newPixels[i * newWidth + itHeightOffset + j + itWidthOffset] = it->pixels[i * itWidth + j];
                            }
                        }
                        //for (int i = 0; i < (maxWidth - minWidth) * (maxHeight - minHeight); i++) {
                        //    if (i % (maxWidth - minWidth) == 0) {
                        //        cout << "\n";
                        //    }
                        //    if (newPixels[i] > LIGHT_GREY)
                        //        cout << "00";
                        //    else {
                        //        cout << "11";
                        //    }

                        //    //cout << (int) newPixels[i];
                        //    /*if (i % (maxWidth - minWidth) == 0) {
                        //        cout << "\n";
                        //    }*/
                        //}
                        // cout << "\n\n";
                        int oldItWidth = oldIt->maxWidth - oldIt->minWidth,
                            oldItHeight = oldIt->maxHeight - oldIt->minHeight,
                            oldItWidthOffset = abs(minWidth - oldIt->minWidth),
                            oldItHeightOffset = abs(minHeight - oldIt->minHeight);
                        for (int i = 0; i < oldItHeight; i++) {
                            for (int j = 0; j < oldItWidth; j++) {
                                /*int q1 = i * itWidth + j,
                                    q2 = i * itWidth + itHeightOffset + j + itWidthOffset,
                                    v1 = it->pixels[i * itWidth + j];*/
                                int offset = itHeight * newWidth + itHeightOffset + itWidth + itWidthOffset;
                                //int q1 = oldIt->pixels[i * oldItWidth + j];
                                //cout << i * newWidth + j + oldItHeightOffset + oldItWidthOffset + itHeight << "\n";
                                newPixels[i * newWidth + j /* + oldItHeightOffset +*/ + oldItWidthOffset + offset] = oldIt->pixels[i * oldItWidth + j];
                                //int q2 = newPixels[i * newWidth + j /* + oldItHeightOffset +*/ + oldItWidthOffset + offset];
                            }
                        }
                        /*it->maxHeight = maxHeight;
                        it->minHeight = minHeight;
                        it->maxWidth = maxWidth;
                        it->minWidth = minWidth;
                        it->pixels = newPixels;*/
                        //for (int i = 0; i < (maxWidth - minWidth) * (maxHeight - minHeight); i++) {
                        //    if (i % (maxWidth - minWidth) == 0) {
                        //        cout << "\n";
                        //    }
                        //    if (newPixels[i] > LIGHT_GREY)
                        //        cout << "00";
                        //    else {
                        //        cout << "11";
                        //    }
                        //    //cout << (int) newPixels[i];
                        //    
                        //}
                        //cout << "\n";
                        //for (int i = 0; i < itWidth * itHeight; i++) {
                        //    if (i % itWidth == 0) {
                        //        cout << "\n";
                        //    }
                        //    if (it->pixels[i] > LIGHT_GREY)
                        //        cout << "00";
                        //    else {
                        //        cout << "11";
                        //    }
                        //    //cout << (int) newPixels[i];

                        //}
                        //cout << "\n";
                        //for (int i = 0; i < oldItWidth * oldItHeight; i++) {
                        //    if (i % oldItWidth == 0) {
                        //        cout << "\n";
                        //    }
                        //    if (oldIt->pixels[i] > LIGHT_GREY)
                        //        cout << "00";
                        //    else {
                        //        cout << "11";
                        //    }
                        //    //cout << (int) newPixels[i];

                        //}
                        // string s;
                        // stringstream ss;
                        // ss << PATH_OUT << IMG_LINE << lineNum << IMG_IDX << letterNum << "two_letters" << IMG_BMP;
                        // ss >> s;
                        // cout << s << "\n";
                        // char* c = const_cast<char*>(s.c_str());
                        // stbi_write_bmp(c, maxWidth - minWidth, maxHeight - minHeight, 1, newPixels /* quality = 100 */);
                        it->minHeight = minHeight;
                        it->maxHeight = maxHeight;
                        it->minWidth = minWidth;
                        it->maxWidth = maxWidth;
                        it->pixels = newPixels;
                        it->letterNum = letterNum - 1;
                        it->lineNum = lineNum;
                        // oldIt = it;
                        // letters2.push_back(it);
                    }
                }
                else {
                    flag = true;
                }
                string s;
                stringstream ss;
                ss << PATH_OUT << IMG_LINE << lineNum << IMG_IDX << letterNum << IMG_BMP;
                ss >> s;
                // cout << s << "\n";
                char* c = const_cast<char*>(s.c_str());
                // stbi_write_bmp(c, it->maxWidth - it->minWidth, it->maxHeight - it->minHeight, 1, it->pixels /* quality = 100 */);
                it->letterNum = letterNum++;
                // letters2.push_back(it);
                oldIt = it;
                it->lineNum = lineNum;
            }
            // cout << "here\n";
            lineNum++;       
            letters.clear();
            letterNum = 0;
        }
    }

    /*
    * Gets the letter image and converts it to a 28 by 28 pixel picture.
    */
    void convertTo28() {
        cout << "aaa\n";
        for (letter_data* it : letters2) {
            int newHeight = it->maxHeight - it->minHeight,
                newWidth  = it->maxWidth  - it->minWidth,
                length = max(newHeight, newWidth),
                offset = (length - newWidth) / 2,
                buffer = (int)(length / 5),
                half_buffer = (int)(buffer / 2 );
            cout << "bbb\n";
            length += buffer;
            uint8_t *sqrePix = new uint8_t[length * length];
                    // *sqrePix_28 = new uint8_t[PIX_28 * PIX_28];
            for (int i = 0; i < length * length; i++) {
                sqrePix[i] = BLACK;
            }
        cout << "ccc\n";

            // for (int i = 0; i < PIX_28 * PIX_28; i++) {
            //     sqrePix_28[i] = BLACK;
            // }
            for (int i = 0; i < newHeight; i++) {
                for (int j = 0; j < newWidth; j++) {
                    sqrePix[(i + half_buffer)* length + j + offset + half_buffer] = it->pixels[i * newWidth + j];
                }
            }
            cout << "dddd\n";
            string s1;
            stringstream ss1;
            if(it->letterNum < 10){
                ss1 << "../resources/out/" << it->lineNum << "_0" << it->letterNum /*<< "not"*/ << IMG_BMP;
            }
            else{
                ss1 << "../resources/out/" << it->lineNum << "_" << it->letterNum /*<< "not"*/ << IMG_BMP;
            }
            ss1 >> s1;
            char* c1 = const_cast<char*>(s1.c_str());
            stbi_write_bmp(c1, length, length, 1, sqrePix);
            // int ratio = (int)(length / PIX_28) + 1,
            //     i = 0, 
            //     j = 0;
            // while (i < length) {
            //     while (j < length){
            //         int sum = 0;
            //         for (int n = 0; n < ratio; n++){
            //             for (int m = 0; m < ratio; m++){
            //                 if(i + n < length && j + m < length){
            //                     sum += sqrePix[(i + n) * length + j + m];
            //                 }
            //             }                    
            //         }
            //         int avg = (int)(sum / (ratio * ratio)),
            //             row_idx = (int)(i / ratio),
            //             col_idx = (int)(j / ratio);
            //         if(row_idx < PIX_28 && col_idx < PIX_28){
            //             sqrePix_28[row_idx * PIX_28 + col_idx] = avg;
            //         }
            //         j += ratio;
            //     }
            //     j = 0;
            //     i += ratio;
            // }
            // i = 0;

            // string s;
            // stringstream ss;
            // if(it->letterNum < 10){
            //     ss << "../resources/out/" << it->lineNum << "_0" << it->letterNum << IMG_BMP;
            // }
            // else{
            //     ss << "../resources/out/" << it->lineNum << "_" << it->letterNum << IMG_BMP;
            // }
            // ss >> s;
            // char* c = const_cast<char*>(s.c_str());
            // stbi_write_bmp(c, PIX_28, PIX_28, 1, sqrePix_28);

            delete[] it->pixels;
            // it->pixels = sqrePix_28;
            delete[] sqrePix;
        }
    }

    void run()
    {
        cout << "here 1\n";
        imgToArray();
        cout << "here 2\n";
        seperateLines();
        cout << "here 3\n";
        seperateLetters();
        cout << "here 4\n";
        convertTo28();
        cout << "here 5\n";

        delete[] image;
        delete[] line_data;
        delete[] image_grey;
        delete[] lines;
        delete[] visitedPixels;

        // return 0;
        //list<letter_data>::iterator it;
        /*for (letter_data* it = letters.front(); it->operator!=(*letters.back()); ++it) {
            delete[] it->pixels;
        }*/
        //delete[] letters;
    };



int main(int argc,char **argv){
    run();
    return 0;
}