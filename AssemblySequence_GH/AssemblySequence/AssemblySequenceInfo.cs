using Grasshopper.Kernel;
using System;
using System.Drawing;

namespace AssemblySequence
{
    public class AssemblySequenceInfo : GH_AssemblyInfo
    {
        public override string Name => "AssemblySequence";

        //Return a 24x24 pixel bitmap to represent this GHA library.
        public override Bitmap Icon => AssemblySequence.Properties.Resources.Assembly_Sequence_icon;

        //Return a short string describing the purpose of this GHA library.
        public override string Description => "This component predicts optimal assembly sequence of a truss using a novel machine learning method combining graph embedding and reinforcement learning.";

        public override Guid Id => new Guid("A5B12EF2-A642-41DA-B7A0-DE9A91F01F8F");

        //Return a string identifying you or your company.
        public override string AuthorName => "Kazuki Hayashi and Makoto Ohsaki (Kyoto University)";

        //Return a string representing your preferred contact details.
        public override string AuthorContact => "hayashi.kazuki.55a@gmail.com";
    }
}