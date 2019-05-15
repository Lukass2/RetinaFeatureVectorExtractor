﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Drawing;
using BIO.Framework.Core;
using BIO.Framework.Extensions.Emgu.InputData;
using BIO.Framework.Extensions.Emgu.FeatureVector;
using BIO.Framework.Extensions.Standard.Template;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;

using BIO.Framework.Extensions.Standard.FeatureVector;
using BIO.Framework.Core.FeatureVector;

namespace BIO.Project.xvejta01.Simple
{
    class RetinaFeatureVectorExtractor3 : IFeatureVectorExtractor<EmguGrayImageInputData, DoubleArrayFeatureVector>
    {
        private int cnt = 1;

        /* SCALE OBRAZKU - zamereni na slepou skvrnu (vcetne translace na podstatna mista) *************************************************************************************************/
        private static Image<Gray, byte> scaleImage2(PointF blindSpot, PointF yellowSpot, Image<Gray, byte> image)
        {
            double Xdiff;      // vzdalenost mezi obema skvrnami v X ose
            double Ydiff;      // vzdalenost mezi obema skvrnami v Y ose
            int PointDiff;  // vzdalenost mezi 2 body
            int X;          // frame origin in x
            int Y;          // frame origin in y
            int Width;      // width of frame
            int Height;     // height of frame

            // ZJISTI VZDALENOST MEZI SKVRNAMI
            // diff x osy
            if (blindSpot.X < yellowSpot.X)
            {
                Xdiff = yellowSpot.X - blindSpot.X;
            }
            else
            {
                Xdiff = blindSpot.X - yellowSpot.X;
            }
            // diff y
            if (blindSpot.Y < yellowSpot.Y)
            {
                Ydiff = yellowSpot.Y - blindSpot.Y;
            }
            else
            {
                Ydiff = blindSpot.Y - yellowSpot.Y;
            }

            // euklidovska vzdalenost mezi obema body
            PointDiff = (int)Math.Round(Math.Sqrt(Math.Pow(Xdiff, 2.0) + Math.Pow(Ydiff, 2.0)));

            // NASTAV POCATECNI POZICI OBRAZKU
            // x
            X = (int)Math.Round(blindSpot.X);
            X -= PointDiff / 2; // posun "kamery" doleva
            // y
            Y = (int)Math.Round(blindSpot.Y);
            Y -= PointDiff / 2; // posun "kamery" nahoru

            // NASTAV ROZMERY VYREZU
            // Width
            Width = 5*PointDiff/3;
            // Height
            Height = Width;

            // ORIZNUTI OBRAZKU NA PATRICNEM MISTE, RESCALE A VRACENI
            image.ROI = new Rectangle(X, Y, Width, Height);
            image = image.Resize(80, 80, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            return image;
        }

        /* SCALE OBRAZKU - zamereni na celou oblast mezi slepou a zlutou skvrnou ***********************************************************************************************************/
        private static Image<Gray, byte> scaleImage(PointF blindSpot, PointF yellowSpot, Image<Gray, byte> image)
        {
            //int Xdiff;// vzdalenost mezi obema skvrnami v X ose
            //int Ydiff;// vzdalenost mezi obema skvrnami v Y ose
            int X;      // frame origin in x
            int Y;      // frame origin in y
            int Width;  // width of frame
            int Height; // height of frame

            // NASTAV POCATECNI POZICI OBRAZKU
            X = (int)Math.Round(blindSpot.X);
            X += 20; // posun "kamery" doprava
            Y = (int)Math.Round(blindSpot.Y);
            Y -= 40; // posun "kamery" nahoru

            // NASTAV ROZMERY VYREZU
            // Width
            if (blindSpot.X < yellowSpot.X)
            {
                Width = (int)Math.Round(yellowSpot.X - blindSpot.X);
            }
            else
            {
                Width = (int)Math.Round(blindSpot.X - yellowSpot.X);
            }
            Width *= 3/2;
            // Height
            Height = 100;

            // ORIZNUTI OBRAZKU NA PATRICNEM MISTE, RESCALE A VRACENI
            image.ROI = new Rectangle(X, Y, Width, Height);
            image = image.Resize(120, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            return image;
        }

        /* ROTACE OBRAZKU (podle obou skvrn) *******************************************************************************************************************/
        private static Image<Gray, byte> rotateImage(PointF blindSpot, PointF yellowSpot, Image<Gray, byte> image)
        {
            float x, y = 0.0F;
            
            if (blindSpot.IsEmpty || yellowSpot.IsEmpty)
            {
                return image;
            }
            // osa x
            if (blindSpot.X < yellowSpot.X)
            {
                x = yellowSpot.X - blindSpot.X;
            }
            else
            {
                x = blindSpot.X - yellowSpot.X;
            }

            float tan = 0.0F;
            // osa y
            if (blindSpot.Y < yellowSpot.Y)
            {
                y = yellowSpot.Y - blindSpot.Y;
            }
            else
            {
                y = blindSpot.Y - yellowSpot.Y;
            }

            // tan = (tan - 2 * tan);
            tan = y / x * (float)(180 / Math.PI);
            if (blindSpot.X < yellowSpot.X)
            {
                if (blindSpot.Y > yellowSpot.Y)
                {
                    //tan = tan;
                }
                else
                {
                    tan = (tan - 2 * tan);
                }
            }
            else
            {
                if (blindSpot.Y > yellowSpot.Y)
                {
                    tan = (tan - 2 * tan);
                }
                else
                {

                }
            }


            image = image.Rotate(tan, new Gray(0));

            /*
            Trace.WriteLine(cnt + " slepá: " + blindSpot.X + "," + blindSpot.Y);
            Trace.WriteLine(cnt + " žlutá: " + yellowSpot.X + "," + yellowSpot.Y);
            Trace.WriteLine(cnt + " tan: " + tan);

            //Trace.WriteLine(" x: " + x + " y: " + y + " z: " + z);
            //Trace.WriteLine("tan: " + tan);
            */

            return image;
        }

        /* DETEKCE KRUHU (slepe a zlute skvrny) ****************************************************************************************************************/
        private static PointF circleCapture(Image<Gray, byte> img, int minColor, int maxColor, int radius, int tolerance)
        {
            /*
            CircleF[] circles = img.HoughCircles(new Gray(255), new Gray(200), 5, 10, 10, 50)[0];
            foreach (CircleF circle in circles) img.Draw(circle, new Gray(255), 2);
            */

            // hleda se slepa skvrna, pokud se nenajde pri danem polomeru, polomer se zmensi a hleda se znovu
            while (radius > 0)
            {
                int radius2 = 2 * radius / 3;   // posun x,y u sikme strany

                // projdeme sloupce a radky + kontrola zda nejsme na okrajich obrazku, 
                // tam skvrna neni (a nedostaneme se mimo indexy pole pri hledani kruhu)
                // sloupce
                for (int y = img.Height / 6; y < (img.Height - img.Height / 6); y++)
                {
                    // radky
                    for (int x = img.Width / 6; x < (img.Width - img.Width / 6); x++)
                    {
                        // vykresleni zpracovavaneho bodu
                        /*
                        PointF center = new PointF(x, y);
                        CircleF circle = new CircleF(center, 0);
                        img.Draw(circle, new Gray(255), 1);
                        //*/

                        // kontrola zda jsme v kruhu (slepa skvrna)
                        int spot = 0;
                        if (img[y, x].Intensity >= minColor && img[y, x].Intensity <= maxColor)
                        {
                            // prosetreni horizontalnich a vertikalnich bodu
                            if (img[y + radius, x].Intensity >= minColor && img[y + radius, x].Intensity <= maxColor) spot++;
                            if (img[y, x + radius].Intensity >= minColor && img[y, x + radius].Intensity <= maxColor) spot++;
                            if (img[y - radius, x].Intensity >= minColor && img[y - radius, x].Intensity <= maxColor) spot++;
                            if (img[y, x - radius].Intensity >= minColor && img[y, x - radius].Intensity <= maxColor) spot++;

                            // prosetreni sikmych bodu
                            if (img[y + radius2, x + radius2].Intensity >= minColor && img[y + radius2, x + radius2].Intensity <= maxColor) spot++;
                            if (img[y + radius2, x - radius2].Intensity >= minColor && img[y + radius2, x - radius2].Intensity <= maxColor) spot++;
                            if (img[y - radius2, x + radius2].Intensity >= minColor && img[y - radius2, x + radius2].Intensity <= maxColor) spot++;
                            if (img[y - radius2, x - radius2].Intensity >= minColor && img[y - radius2, x - radius2].Intensity <= maxColor) spot++;

                            // prosetreni horizontalnich a vertikalnich vnitrnejsich bodu
                            if (img[y + radius / 2, x].Intensity >= minColor && img[y + radius / 2, x].Intensity <= maxColor) spot++;
                            if (img[y, x + radius / 2].Intensity >= minColor && img[y, x + radius / 2].Intensity <= maxColor) spot++;
                            if (img[y - radius / 2, x].Intensity >= minColor && img[y - radius / 2, x].Intensity <= maxColor) spot++;
                            if (img[y, x - radius / 2].Intensity >= minColor && img[y, x - radius / 2].Intensity <= maxColor) spot++;

                            // nasli jsme slepou skvrnu? jestli ano, ukonceni hledani
                            PointF center = new PointF(x, y);
                            CircleF circle = new CircleF(center, radius);
                            if (spot >= tolerance)
                            {
                                //img.Draw(circle, new Gray(125), 1); //???
                                return center;
                            }
                        }
                    }
                }
                // zmensime radius o desetinu a zkusime skvrnu najit znovu
                radius = radius - 1;
            }   // end of while

            // NENALEZENO
            //return circleCapture(img, minColor, maxColor, 1, 1);
            return new PointF(0, 0);
        }



        /* *****************************************************************************************************************************************************/
        /* ULOZENI MARKANTU ************************************************************************************************************************************/
        private static DoubleArrayFeatureVector imageReturnVector(Image<Gray, byte> image)
        {
            DoubleArrayFeatureVector fv = new DoubleArrayFeatureVector(image.Width * image.Height);

            // ULOZENI VEKTORU MARKANTU
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int index = y * image.Width + x;
                    fv.FeatureVector[index] = image[y, x].Intensity;
                }
            }

            // NAVRACENI VEKTORU
            return fv;
        }

        /* ZVYRAZNENI MARKANTU *********************************************************************************************************************************/
        private static Image<Gray, byte> imageFeatures(Image<Gray, byte> image)
        {
            // EXTRAHOVANI ZIL
            //smaller = smaller.Canny(new Gray(100), new Gray(255));
            //smaller._ThresholdBinary(new Gray(50), new Gray(255));
            //smaller = smaller.Resize(0.2, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

            // EKVALIZACE HISTOGRAMU
            //image._EqualizeHist();
            
            // VYHLAZENI
            //image = image.SmoothGaussian(5, 5, 1.5, 1.5);
            
            //image._GammaCorrect(5);

            // NALEZENI HRAN
            //image = image.Canny(new Gray(1), new Gray(1));

            // DALSI OPRAVY
            //image = image.Dilate(1);
            //image = image.Erode(1);
            //image = image._ThresholdBinary(new Gray(240), new Gray(255));
            //smaller._ThresholdTrunc(new Gray(150));

            // NAVRACENI OBRAZU
            return image;
        }

        /* RESHAPE OBRAZU **************************************************************************************************************************************/
        private static Image<Gray, byte> imageReshape(Image<Gray, byte> image, int cnt)
        {
            PointF blindSpot;
            PointF yellowSpot;
            int minColor;                   // podstatne pro urceni presneho kolecka, nebo i pred tim binarizovat obraz a pod
            int maxColor;
            int radius;                     // musi byt mensi jak img.Height/6, tedy /7, /8, ...
            int tolerance;
            
            // NALEZENI SKVRN
            // nalezeni žluté skvrny (cerna)
            minColor = 0;
            maxColor = 25;
            radius = image.Height / 6;
            tolerance = 11;
            yellowSpot = circleCapture(image, minColor, maxColor, radius, tolerance);
            // nalezeni slepé skvrny (bila)
            minColor = 245;
            maxColor = 255;
            radius = image.Height / 6;
            tolerance = 9;
            blindSpot = circleCapture(image, minColor, maxColor, radius, tolerance);

            // DEBUGOVANI
            //CvInvoke.cvSaveImage("C:/Download/img/done/" + cnt.ToString() + "-spots.png", image.Ptr);       // ulozeni

            // ROTACE - podle stredu obou skvrn
            image = rotateImage(blindSpot, yellowSpot, image);           

            // SCALE + TRANSLACE
            image = scaleImage2(blindSpot, yellowSpot, image);

            // DEBUGOVANI
            //CvInvoke.cvSaveImage("C:/Download/img/done/" + cnt.ToString() + "-RotSc.png", image.Ptr);       // ulozeni

            // NAVRACENI OBRAZU
            return image;
        }

        /* FITRACE OBRAZU **************************************************************************************************************************************/
        private static Image<Gray, byte> imageFilter(EmguGrayImageInputData input)
        {
            // RESIZE A VYTVORENI FORMY OBRAZKU
            Image<Gray, byte> image = input.Image.Resize(0.2, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

            // EKVALIZACE HISTOGRAMU
            image = image.SmoothGaussian(5, 5, 1.5, 1.5);
            image._EqualizeHist();

            // NAVRAT FILTROVANEHO OBRAZKU
            return image;
        }

        /* OSETRENI VSTUPNIHO FORMATU **************************************************************************************************************************/
        void imageSize(EmguGrayImageInputData input)
        {
            // KONTROLA VSTUPNI VELIKOSTI
            //if (input.Image.Width != 768 || input.Image.Height != 584) throw new InvalidOperationException("Image size has to be 768x584 pixels.");
        }



        /* *****************************************************************************************************************************************************/
        /* FEATURE EXTRACTOR ***********************************************************************************************************************************/
        #region IFeatureVectorExtractor<EmguGrayImageInputData,DoubleArrayFeatureVector> Members
        public DoubleArrayFeatureVector extractFeatureVector(EmguGrayImageInputData input)
        {
            Image<Gray, byte> image;

            // DEBUGOVANI
            //CvInvoke.cvSaveImage("C:/Download/img/done/" + cnt.ToString() + ".png", input.Image.Ptr);       // ulozeni

            // OSETRENI VSTUPU
            imageSize(input);

            // FILTRACE
            image = imageFilter(input);

            // OPRAVA ROTACE, SCALU A TRANSLACE
            image = imageReshape(image, cnt);
            
            // ZVYRAZNENI MARKANTU V OBRAZE
            image = imageFeatures(image);//???
            
            // DEBUGOVANI
            //CvInvoke.cvShowImage("ALG3", image.Ptr);                                                // vykresleni
            //CvInvoke.cvSaveImage("C:/Download/img/done/" + cnt.ToString() + "-done.png", image.Ptr);       // ulozeni
            cnt++;

            // NAVRACENI VEKTORU MARKANTU
            return imageReturnVector(image);
        }

        #endregion
    }
}
