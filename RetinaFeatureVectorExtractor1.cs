using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
    class RetinaFeatureVectorExtractor1 : IFeatureVectorExtractor<EmguGrayImageInputData, DoubleArrayFeatureVector>
    {
        private int cnt = 1;

        #region IFeatureVectorExtractor<EmguGrayImageInputData,DoubleArrayFeatureVector> Members
        public DoubleArrayFeatureVector extractFeatureVector(EmguGrayImageInputData input)
        {
            // OSETRI VSTUPNI OBRAZEK
            //if (input.Image.Width != 767 || input.Image.Height != 583) throw new InvalidOperationException("Image size has to be 767x583 pixels.");

            // UPRAV OBRAZEK
            Image<Gray, byte> smaller = input.Image.Resize(0.2, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            smaller._SmoothGaussian(3);
            smaller._EqualizeHist();

            // ULOZ FEATURE VEKTOR
            //CvInvoke.cvShowImage("ALG2", smaller.Ptr);
            //CvInvoke.cvSaveImage("C:/Download/img/equa/" + cnt.ToString() + ".png", smaller.Ptr);     // ulozeni
            cnt++;

            DoubleArrayFeatureVector fv = new DoubleArrayFeatureVector(smaller.Width * smaller.Height);
            for (int y = 0; y < smaller.Height; y++)
            {
                for (int x = 0; x < smaller.Width; x++)
                {
                    int index = y * smaller.Width + x;
                    fv.FeatureVector[index] = smaller[y, x].Intensity;
                }
            }

            return fv;
        }

        #endregion
    }
}
