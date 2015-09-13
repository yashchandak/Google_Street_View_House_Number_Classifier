/** TODO :
//MSER (int) in loops??
//MSER hierarchy
//make MSER values dynamic//DO 5 ROUNDS
    contour on only edge (if inner heirarchy present then dont use current edge contour
    //one with edge convolution
    other without toggle
    //third with disable
    MSER with and without binary INV

    OTSU threshold of MSER given regions

//remove overlapping windows
//combine Top and bottom together
//split in left and right if two digits are combined together

//store the bounding boxes of image in a text file.
//classify individual digits and store accordingly in 10 separate files representing each number.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include<stdio.h>
#include<iostream>
#include<fstream>

using namespace cv;
using namespace std;


class getbbox
{
public:

    Mat ycrcb_channels[3];
    Mat ycrcb, thresh, thresh_inv, image;
    Mat Gthresh, Gthresh_inv;
    Mat edge;
    Mat cropped;
    int kSize, totalPx;
    float Kfactor, minFactor, maxFactor; //factor for thresholding kernel size based on image size
    Mat mark, result, result2;

    fstream out;
    ostringstream ss;
    int cc, imNo;
    string filename;

    int minSize, maxSize;
    float ratio;
    float  contour_area, rect_area;
    float bounding_width;
    float bounding_length;
    vector< vector<Point> > contours;
    vector< vector<Point> > edge_contours;
    vector< vector<Point> > contours_inv;
    vector<Vec4i> hierarchy;
    vector<Vec4i> edge_hierarchy;
    vector<Vec4i> hierarchy_inv;
    Rect bounded_rect;
    vector<Rect> rectList;
    vector<Rect> mergedList;

    //variables for drawBbox()
    Mat bw[3];
    Mat input;
    Point topLeft, bottomRight, temp;
    Mat th, thI;
    float expand;
    vector< vector<Point> > inner_contours;
    vector< vector<Point> > inner_contours_inv;
    vector<Vec4i> inner_hierarchy;
    vector<Vec4i> inner_hierarchy_inv;
    int inner_minSize;
    int inner_maxSize;
    int center_x;

    //variables for MSER
    Mat img, yuv, gray;
    vector<vector<Point> > mser_contours;
    vector<Vec4i>mser_hierarchy;

    getbbox(float num=0.006, float num1=0.005, float num2=0.5, float num3 = 0.1)
    {
        namedWindow("original",2);
        namedWindow("thresholded",2);
        namedWindow("thresholded_Inv",2);
        namedWindow("final",2);
        namedWindow("final2",2);
        namedWindow("edge",2);
        namedWindow("th",2);
        namedWindow("thI",2);
        namedWindow("binary",2);
        namedWindow("cropped",2);
        namedWindow("background",2);
        namedWindow("original",2);
        namedWindow("response",2);
        namedWindow("final3",2);
        //namedWindow("y",2);
        //namedWindow("cr",2);
        //namedWindow("cb",2);

        //image invariant parameters
        Kfactor = num;
        minFactor = num1;
        maxFactor = num2;
        expand = num3;
        ratio = 2;

        cc=0;
        imNo =0;
        out.open("location.txt", ios::out);
    }

    void setParam()
    {
        totalPx = image.rows*image.cols;
        kSize = totalPx*Kfactor;
        kSize = (kSize/2)*2 +1;
        minSize = minFactor*totalPx;
        maxSize = maxFactor*totalPx ;

        contours.clear();
        contours_inv.clear();

        hierarchy.clear();
        hierarchy_inv.clear();

        edge_contours.clear();
        edge_hierarchy.clear();

        mser_contours.clear();
        mser_hierarchy.clear();

        inner_contours.clear();
        inner_hierarchy.clear();

        inner_contours_inv.clear();
        inner_hierarchy_inv.clear();

        rectList.clear();
        mergedList.clear();

        cout<<"r: "<<image.rows<<" c: "<<image.cols<<" k: "<<kSize<<" min: "<<minSize<<" max : "<<maxSize<<endl;
    }

    void getEdge()
    {
        Canny(ycrcb_channels[0], edge, 50, 150);
        //imshow("edge",edge);
    }

    void edgeContour()
    {
        Mat temp;
        edge.copyTo(temp);

        findContours(temp, edge_contours, edge_hierarchy, CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);

        // for removing invalid blobs
        if (!edge_contours.empty())
        {
            for (size_t i=0; i<edge_contours.size(); ++i)
            {
                //doesnt have any inner contour nor has any child
                if(edge_hierarchy[i][3]==-1 || edge_hierarchy[i][2] == -1)
                {
                    //====conditions for removing edge_contours====//
                    bounded_rect    = boundingRect(edge_contours[i]);
                    bounding_width  = bounded_rect.width;
                    bounding_length = bounded_rect.height;

                    contour_area = contourArea(edge_contours[i]) ;
                    rect_area = bounding_length*bounding_width;

                    //blob size should be more than lower threshold
                    int center_x = bounded_rect.tl().x + bounded_rect.width/2;

                    if(center_x> image.cols/5 && center_x<(image.cols*4)/5)
                    {
                        //if(bounding_length < 0.8*image.rows &&((contour_area >= minSize && contour_area <= maxSize)))
                        {
                            if ((bounding_width/bounding_length) <= ratio)
                            {
                                drawBox(bounded_rect, Scalar(0,255,255));//without inverse
                                drawBox(bounded_rect, Scalar(0,255,255),1);//with inverse
                                rectangle(result2, bounded_rect, Scalar(0,255,255), 1, 8, 0 );
                                //drawContours(blob, edge_contours, i, Scalar(255));
                            }

                        }
                    }
                }
            }
        }

    }

    Mat convoluteEdge(Mat& src, int toggle = 0)
    {
        for(int i =0; i<src.rows; i++)
        {
            for(int j = 0; j<src.cols; j++)
            {
                if(edge.at<uchar>(i,j) == 255)
                {
                    if(toggle == 1 )
                        src.at<uchar>(i,j) = (src.at<uchar>(i,j)==255)?0 : 255;
                    else
                        src.at<uchar>(i,j) = (src.at<uchar>(i,j)==255)?0 : src.at<uchar>(i,j);
                }
            }
        }

        return src;
    }

    void drawBox( Rect bounded_rect, Scalar color, int inverted = 0)
    {

        topLeft.x = std::max(0, int(bounded_rect.tl().x - 0.1*bounded_rect.width));
        topLeft.y = std::max(0, int(bounded_rect.tl().y - 0.1*bounded_rect.height)) ;

        bottomRight.x = std::min(result.cols, int(bounded_rect.br().x + 0.1*bounded_rect.width));
        bottomRight.y = std::min(result.rows, int(bounded_rect.br().y + 0.1*bounded_rect.height));

        Rect myROI(topLeft, bottomRight);
        cropped = image(myROI);//.clone();

        cvtColor(cropped, input, CV_RGB2GRAY,1);
        split(input, bw);
        threshold(bw[0],th,0,255,CV_THRESH_OTSU);
        bitwise_not ( th, thI );

        imshow("th",th);
        imshow("thI", thI);

        inner_minSize = 0.005*(myROI.height*myROI.width);//0.005
        inner_maxSize = 0.80*(myROI.height*myROI.width);//0.75
        //inner_minSize = 0.005*(bounded_rect.height*bounded_rect.width);//0.005
        //inner_maxSize = 0.80*(bounded_rect.height*bounded_rect.width);//0.75

        if(inverted == 0)
        {
            findContours(th, inner_contours, inner_hierarchy, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
            if (!inner_contours.empty())
            {
                for (size_t i=0; i<inner_contours.size(); ++i)
                {
                    if(inner_hierarchy[i][3]==-1)
                    {
                        //====conditions for removing contours====//
                        bounded_rect    = boundingRect(inner_contours[i]);
                        bounding_width  = bounded_rect.width;
                        bounding_length = bounded_rect.height;
                        contour_area = contourArea(inner_contours[i]) ;
                        rect_area = bounding_length*bounding_width;

                        center_x = bounded_rect.tl().x + bounded_rect.width/2;
                        //contour area and rectangular area respectively
                        if(contour_area >= inner_minSize && (rect_area <= inner_maxSize || contour_area >= (0.95 * rect_area)))
                        {
                            if ((bounding_width/bounding_length) <= ratio && (bounding_length/bounding_width)<10)
                            {
                                bounded_rect.x += topLeft.x;
                                bounded_rect.y += topLeft.y;
                                rectangle(result, bounded_rect , color, 1, 8, 0 );
                                rectList.push_back(Rect(bounded_rect));
                            }
                        }

                    }

                }
            }
        }
        else
        {
            findContours(thI, inner_contours_inv, inner_hierarchy_inv, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
            if (!inner_contours_inv.empty())
            {
                for (size_t i=0; i<inner_contours_inv.size(); ++i)
                {
                    if(inner_hierarchy_inv[i][3]==-1)
                    {
                        //====conditions for removing contours====//
                        bounded_rect   =  boundingRect(inner_contours_inv[i]);
                        bounding_width  = bounded_rect.width;
                        bounding_length = bounded_rect.height;
                        rect_area = bounding_length*bounding_width;

                        contour_area = contourArea(inner_contours_inv[i]) ;

                        if(contour_area >= inner_minSize && (rect_area <= inner_maxSize || contour_area >= (0.95 * rect_area)))
                        {
                            if ((bounding_width/bounding_length) <= ratio  && (bounding_length/bounding_width)<10)
                            {
                                bounded_rect.x += topLeft.x;
                                bounded_rect.y += topLeft.y;
                                rectangle(result, bounded_rect , color, 1, 8, 0 );
                                rectList.push_back(Rect(bounded_rect));
                            }
                        }
                    }
                }
            }
        }

        //imshow("binary",bw[0]);
        //imshow("cropped", cropped);
        imshow("final", result);

        //waitKey(0);
    }

    void perform_MSER(Mat img0)
    {
        Vec3b bcolors[] =
        {
            Vec3b(0,0,255),
            Vec3b(0,128,255),
            Vec3b(0,255,255),
            Vec3b(0,255,0),
            Vec3b(255,128,0),
            Vec3b(255,255,0),
            Vec3b(255,0,0),
            Vec3b(255,0,255),
            Vec3b(255,255,255)
        };
        // ==== FAILS FOR BIG WHITE REGIONS ====

        cvtColor(img0, yuv, COLOR_BGR2YCrCb);
        cvtColor(img0, gray, COLOR_BGR2GRAY); //useless
        cvtColor(gray, img, COLOR_GRAY2BGR);

        mark = Mat(img0.rows, img0.cols, CV_8UC1, 255);

        //medianBlur(img0, img0, 3); //(?)

        MSER(10,200,200000,0.4,0.4,300)(img0, mser_contours);
        //MSER(10,100,200000,0.2,0.4,300)(img0, contours);

        // draw MSERs with different colors
        for( int i = (int)mser_contours.size()-1; i >= 0; i-- )
        {
            const vector<Point>& r = mser_contours[i];
            for ( int j = 0; j < (int)r.size(); j++ )
            {
                Point pt = r[j];
                //img.at<Vec3b>(pt) = bcolors[i%9];//coloring wont be required later, remove it.
                mark.at<uchar>(pt) = 0;
            }
        }

        imshow( "background", mark );

        findContours(mark, mser_contours,mser_hierarchy, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);

        if (!mser_contours.empty())
        {
            for (size_t i=0; i<mser_contours.size(); ++i)
            {
                //====conditions for removing mser_contours====//
                bounded_rect    = boundingRect(mser_contours[i]);
                bounding_width  = bounded_rect.width;
                bounding_length = bounded_rect.height;
                contour_area = contourArea(mser_contours[i]) ;
                rect_area = bounding_length*bounding_width;

                int center_x = bounded_rect.tl().x + bounded_rect.width/2;

                //exploiting property of dataset, most numbers are almost in centre
                if(center_x> image.cols/5 && center_x<(image.cols*4)/5)
                {
                    if(contour_area >= minSize && contour_area <= maxSize)
                    {
                        //if ((bounding_width/bounding_length) <= ratio)
                        {
                            drawBox(bounded_rect, Scalar(255,0,0));//without inverse
                            drawBox(bounded_rect, Scalar(255,0,0),1);//with inverse
                            rectangle(result2, bounded_rect, Scalar(255,0,0), 1, 8, 0 );
                        }
                    }
                }
            }
        }

        //imshow( "original", img0 );
        imshow( "response", img );
        imshow("final2", result2);

    }

    void getBinary(Mat image)
    {
        Mat binary[3];
        Mat input;

        cvtColor(image, input, CV_RGB2GRAY,1);
        split(input, binary);
        threshold(binary[0],binary[0],0,255,CV_THRESH_OTSU);

        /** PREPROCESSING FOR SENDING IT TO CLASSIFIER
        int erosion_size=1;
          Mat element = getStructuringElement( MORPH_RECT,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
          Apply the erosion operation
          erode( binary[0], binary[0], element);
           //blur(image, image,Size(7,7));*/

        namedWindow("binary",2);
        imshow("binary",binary[0]);

    }

    void mergeBbox()
    {
        //groupRectangles(rectList, 0 , 10);
        Mat result3;
        Rect rect1, rect2;
        image.copyTo(result3);
        int *marked;
        marked = new int[rectList.size()];

        for(int i=0; i<rectList.size(); i++)
        {
          marked[i]=0;
          rectangle(result2, rectList[i] , Scalar(255, 0 ,255), 1, 8, 0 );
        }

        imshow("final2", result2);

        for(int i=0; i<rectList.size(); i++)
        {
            rect1 = rectList[i];
            if (marked[i]==0)
            {
                for(int j = i; j<rectList.size(); j++)
                {
                    if(marked[j]==0)
                    {
                        rect2 = rectList[j];
                        if((rect1 & rect2) == rect1)//rect1 completely belongs to rect2
                        {
                            rect1 = rect2;
                            marked[j]=1;
                        }
                        else if((rect1 & rect2) == rect2) // rect2 is completely inside rect1
                        {
                            marked[j]=1;
                        }
                        else if((rect1 & rect2).area() > 0.5*rect1.area() ||(rect1 & rect2).area() > 0.5*rect2.area() ) // they intersect more than half; merge them.
                        {
                            rect1 = rect1 | rect2;
                            marked[j]=1;
                        }
                    }
                }
                mergedList.push_back(rect1);
            }
        }

        for(int i=0; i<mergedList.size(); i++)
                rectangle(result3, mergedList[i] , Scalar(255, 255 ,255), 1, 8, 0 );

        imshow("final3", result3);
       // del [] marked;

    }

    void bbox()
    {
        /**IMP findcontours modifies source matrix**/
        findContours(thresh, contours,hierarchy, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
        findContours(thresh_inv, contours_inv,hierarchy_inv, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);

        Mat blob(thresh.rows, thresh.cols, CV_8UC1, 0.0);
        Mat blob_inv(thresh.rows, thresh.cols, CV_8UC1, 0.0);
        image.copyTo(result);
        image.copyTo(result2);
        namedWindow("blob",2);
        namedWindow("blob_inv",2);

        // for removing invalid blobs
        if (!contours.empty())
        {
            for (size_t i=0; i<contours.size(); ++i)
            {
                if(hierarchy[i][3]==-1)
                {
                    //====conditions for removing contours====//
                    bounded_rect    = boundingRect(contours[i]);
                    bounding_width  = bounded_rect.width;
                    bounding_length = bounded_rect.height;

                    contour_area = contourArea(contours[i]) ;
                    rect_area = bounding_length*bounding_width;

                    //blob size should be more than lower threshold
                    int center_x = bounded_rect.tl().x + bounded_rect.width/2;

                    if(center_x> image.cols/5 && center_x<(image.cols*4)/5)
                    {
                        if(bounding_length < 0.8*image.rows &&((contour_area >= minSize && contour_area <= maxSize)) && (contour_area < 0.80*rect_area))
                        {
                            if ((bounding_width/bounding_length) <= ratio)
                            {
                                drawBox(bounded_rect, Scalar(0,255,0));
                                rectangle(result2, bounded_rect, Scalar(0,255,0), 1, 8, 0 );
                                drawContours(blob, contours, i, Scalar(255));
                            }
                            /*cout<<rect_area<<" "<<contour_area<<" "<<bounding_width<<" "<<bounding_length<<endl;
                            imshow("blob",blob);
                            imshow("final", result);*/
                            //waitKey(0);

                        }
                    }
                }
            }
        }
        if (!contours_inv.empty())
        {
            for (size_t i=0; i<contours_inv.size(); ++i)
            {
                if(hierarchy_inv[i][3]==-1)
                {
                    //====conditions for removing contours====//
                    bounded_rect   =  boundingRect(contours_inv[i]);
                    bounding_width  = bounded_rect.width;
                    bounding_length = bounded_rect.height;

                    contour_area = contourArea(contours_inv[i]) ;
                    rect_area = bounding_length*bounding_width;

                    int center_x = bounded_rect.tl().x + bounded_rect.width/2;

                    if(center_x> image.cols/5 && center_x<(image.cols*4)/5)
                    {
                        //blob size should be more than lower threshold
                        if(bounding_length < 0.8*image.rows &&(contour_area >= minSize && contour_area <= maxSize) && (contour_area < 0.80*rect_area))
                        {

                            if ((bounding_width/bounding_length) <= ratio)
                            {
                                drawBox(bounded_rect, Scalar(0,0,255),1);
                                rectangle(result2, bounded_rect, Scalar(0,0,255), 1, 8, 0 );
                                drawContours(blob_inv, contours_inv, i, Scalar(255));
                            }
                            /*cout<<rect_area<<" "<<contour_area<<" "<<bounding_width<<" "<<bounding_length<<endl;
                            imshow("blob_inv",blob_inv);
                            imshow("final", result);*/
                            //waitKey(0);
                        }
                    }
                }
            }
        }

        imshow("final2", result2);
        imshow("blob",blob);
        imshow("blob_inv",blob_inv);
    }

    void nextImage(Mat img)
    {
        imNo++;
        image = img;
        imshow("original",image);
        setParam();

        cvtColor(image, ycrcb, CV_RGB2YCrCb);
        split(ycrcb, ycrcb_channels);

        //imshow("y",ycrcb_channels[0]);
        //imshow("cr", ycrcb_channels[1]);
        //imshow("cb", ycrcb_channels[2]);

        adaptiveThreshold(ycrcb_channels[0], thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,kSize,10);
        bitwise_not ( thresh, thresh_inv );
        //adaptiveThreshold(ycrcb_channels[0], thresh_inv, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,kSize,10);

        //separating wrongly joint blobs using edges
        getEdge();
        convoluteEdge(thresh,1);
        convoluteEdge(thresh_inv,1);

        imshow("thresholded", thresh);
        imshow("thresholded_Inv", thresh_inv);
        bbox();
    }

    void write()
    {
        out<<imNo<<" "<<mergedList.size();
        for(int i =0; i<mergedList.size(); i++)
            out<<" "<<mergedList[i].x<<" "<<mergedList[i].y<<" "<<mergedList[i].width<<" "<<mergedList[i].height;
        out<<"\n";
    }
    ~getbbox()
    {
        out.close();
    }

};


int main()
{
    //READ color image, check for other types not provided yet
    Mat image;// = imread("D:/ToDo/research_santa_cruz/train/22.png",1);//7458,359,421,16;452 598!! ;154
    //imread("test2.png",1);
    getbbox SVHN;
    string address = "D:/ToDo/research_santa_cruz/train/";
    string filename;
    for(int i =400; i<500 ; i++)
    {
        cout<<i<<endl;
        ostringstream ss;
        ss<<i;
        //cout<<"address : "<<address<<endl;
        filename = address + ss.str() + ".png";
        cout<<filename<<endl;
        image = imread(filename,1);

        SVHN.nextImage(image);
        SVHN.perform_MSER(image);
        //SVHN.edgeContour();
        SVHN.mergeBbox();
        SVHN.write();

        waitKey(0);
    }

    waitKey(0);
}




//double t = (double)getTickCount();
// --- CODE ---
//t = (double)getTickCount() - t;
//printf( "MSER extracted %d contours in %g ms.\n", (int)contours.size(),t*1000./getTickFrequency() );

/*

    void rgbThresh(Mat image)
    {

        Mat bgr[3];
        split(image,bgr);
        Mat bThresh, gThresh, rThresh;

        adaptiveThreshold(bgr[0], bThresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,kSize,10);
        adaptiveThreshold(bgr[1], gThresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,kSize,10);
        adaptiveThreshold(bgr[2], rThresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,kSize,10);

        namedWindow("rThresh",2);
        namedWindow("gThresh",2);
        namedWindow("bThresh",2);

        imshow("gThresh", gThresh);
        imshow("bThresh", bThresh);
        imshow("rThresh", rThresh);
    }



        /*ss<<cc++;
        //cout<<"address : "<<address<<endl;
        filename = ss.str() + ".png";
        imwrite(filename, th);
         ss<<cc++;
        //cout<<"address : "<<address<<endl;
        filename = ss.str() + ".png";
        imwrite(filename, thI);
    */


    /** morphological operator
        // Opening: MORPH_OPEN : 2
         //Closing: MORPH_CLOSE: 3
         //Gradient: MORPH_GRADIENT: 4
         //Top Hat: MORPH_TOPHAT: 5
         //Black Hat: MORPH_BLACKHAT: 6

         int morph_size =1;
         Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

         /// Apply the specified morphology operation
         // morphologyEx( ycrcb_channels[0], ycrcb_channels[0], 2, element );
         //morphologyEx( ycrcb_channels[0], ycrcb_channels[0], 3, element );
         //imshow( window_name, dst );
         **/

