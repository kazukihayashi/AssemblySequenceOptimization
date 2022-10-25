using GraphEmbedding;
using Grasshopper.Kernel;
using Numpy;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AssemblySequence
{

    public class AssemblySequenceComponent : GH_Component
    {
        static Rhino.Collections.Point3dList points;
        static List<Point3d> points_list = new List<Point3d>();
        static List<Line> lines_list = new List<Line>();
        static List<Point3d> supports = new List<Point3d>();
        static List<int>[] connected_members;
        static NDarray v;
        static NDarray w;
        static NDarray<int> c;
        static NDarray c_temp;
        static int dim;
        static int nn;
        static int nm;
        static Agent agent = new Agent(2, 0, 100);
        static NDarray coords;
        static NDarray<int> spi;
        static NDarray exist_member_i;
        static int a;
        static int[] order;
        static NDarray<bool> node_existence;
        static bool[] node_stable;
        static CancellationTokenSource tokenSource = new CancellationTokenSource();
        static CancellationToken token;
        private static Task<int[]> task;
        public int Valence { get; private set; }

        /// <summary>
        /// Each implementation of GH_Component must provide a public 
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear, 
        /// Subcategory the panel. If you use non-existing tab or panel names, 
        /// new tabs/panels will automatically be created.
        /// </summary>
        public AssemblySequenceComponent()
          : base("AssemblySequence", "AS",
            "Output sub-optimal assembly sequence of bar elements.",
            "Extra", "Extra")
        {
        }

        /// Add time display below the component
        private void UpdateDisplay(in string s)
        {
            Message = s;
            Grasshopper.Instances.InvalidateCanvas();
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Point", "P", "Truss nodes", GH_ParamAccess.list);
            pManager.AddLineParameter("Line", "L", "Truss bars", GH_ParamAccess.list);
            pManager.AddPointParameter("Support", "SP", "Pin supports", GH_ParamAccess.list);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Order", "O", "Assembly order of bar member", GH_ParamAccess.list);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        /// 
        protected override void BeforeSolveInstance()
        {
            Valence = 0;
            GH_Document doc = OnPingDocument();
            if (doc != null)
                foreach (IGH_DocumentObject obj in doc.Objects)
                    if (obj.ComponentGuid == this.ComponentGuid)
                        Valence++;
            if (Valence > 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Only a single component of this type is allowed per file.");
            }

            if (task != null)
            {
                tokenSource.Cancel();
                task.Wait();
                tokenSource = new CancellationTokenSource();
            }
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            if (Valence > 1)
            {
                return;
            }

            points_list.Clear();
            lines_list.Clear();
            supports.Clear();

            if (!DA.GetDataList(0, points_list)) return;
            if (!DA.GetDataList(1, lines_list)) return;
            if (!DA.GetDataList(2, supports)) return;

            System.Diagnostics.Stopwatch time = new System.Diagnostics.Stopwatch();
            time.Start();

            points = new Rhino.Collections.Point3dList(points_list);

            for (int i = 0; i < supports.Count; i++)
            {
                if (!points.Contains(supports[i]))
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Support point {0} is not included in truss nodes.");
                }
            }

            nn = points_list.Count;
            nm = lines_list.Count;

            coords = np.zeros(nn, 3);
            for (int i = 0; i < points_list.Count; i++)
            {
                coords[i, 0] = (NDarray)points_list[i].X;
                coords[i, 1] = (NDarray)points_list[i].Y;
                coords[i, 2] = (NDarray)points_list[i].Z;
            }

            dim = np.linalg.matrix_rank(coords);
            if (dim < 3)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "This truss is not a 3D model. This may cause inaccuracy in estimating the assembly sequence, because the machine learning model is trained for 3D models.");
            }

            c = np.asarray(new int[nm, 2]);
            for (int i = 0; i < nm; i++)
            {
                c[i, 0] = (NDarray)Rhino.Collections.Point3dList.ClosestIndexInList(points, lines_list[i].From);
                c[i, 1] = (NDarray)Rhino.Collections.Point3dList.ClosestIndexInList(points, lines_list[i].To);
            }
            c.sort(1);
            c_temp = np.copy(c);

            spi = np.asarray(new int[supports.Count]);
            for (int i = 0; i < supports.Count; i++)
            {
                spi[i] = (NDarray)Rhino.Collections.Point3dList.ClosestIndexInList(points, supports[i]);
            }

                
            connected_members = Connected_Members(nn, c);
            node_stable = Nodal_Stability(coords, c, connected_members);
            v = Set_v(spi, node_stable);
            w = np.zeros(nm, 1);
            exist_member_i = np.arange(nm);
            node_existence = Node_Exist(nn, c);
                
            token = tokenSource.Token;
            task = Task.Run(() => Iterate(token));
            order = task.Result;

            time.Stop();
            if (!token.IsCancellationRequested)
            {
                UpdateDisplay(string.Format("Elapsed time: {0}", time.Elapsed.TotalSeconds));
                DA.SetDataList(0, order);
            }
            task = null;
        }

        int[] Iterate(CancellationToken token)
        {
            int[] od = new int[nm];
            for (int i = 0; i < nm; i++)
            {
                if (!token.IsCancellationRequested)
                {
                    a = agent.Decide(v[node_existence], w[exist_member_i], c_temp);
                    int a_to_m = (int)exist_member_i[a];
                    od[i] = a_to_m;
                    exist_member_i = np.delete(exist_member_i, a);

                    for (int j = 1; j > -1; j--)
                    {
                        connected_members[(int)c[a_to_m, j]].Remove(a_to_m);
                        if (connected_members[(int)c[a_to_m, j]].Count == 0)
                        {
                            node_existence[c[a_to_m, j]] = (NDarray)false;
                            c_temp[c_temp > c_temp[a, j]] -= 1;
                        }
                        node_stable = Nodal_Stability(coords, c, connected_members, node_stable, (int)c[a_to_m, j]);
                    }
                    c_temp = np.delete(c_temp, a, 0);

                    v = Set_v(spi, node_stable, v);
                    UpdateDisplay(string.Format("Iteration {0}/{1}", i, nm));
                }
                else
                {
                    UpdateDisplay(string.Format("Cancelled"));
                    return null;
                }
            }
            return od;
        }

        List<int>[] Connected_Members(int nn, NDarray c)
        {
            connected_members = new List<int>[nn].Select(item => new List<int>()).ToArray();
            for (int i = 0; i < nm; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    connected_members[(int)c[i, j]].Add(i);
                }
            }
            return connected_members;
        }

        NDarray Set_v(NDarray<int> spi, bool[] node_stable, NDarray v = null)
        {
            if (v == null)
            {
                v = np.zeros(nn, 2);
                v[spi, 0] = (NDarray)1.0;
            }
            v[np.logical_not(node_stable),1] = (NDarray)1.0;
            v[spi, 1] = (NDarray)0.0;
            return v;
        }

        NDarray<bool> Node_Exist(int nn_temp, NDarray c_temp)
        {
            NDarray<bool> node_existence = np.asarray(new bool[nn_temp]);
            node_existence[c_temp.flatten()] = (NDarray)true;
            return node_existence;
        }

        bool[] Nodal_Stability(NDarray coords, NDarray c, List<int>[] connected_members)
        {
            bool[] node_stable = Enumerable.Repeat(true, nn).ToArray();
            for (int i = 0; i < nn; i++)
            {
                if (connected_members[i].Count < dim)
                {
                    node_stable[i] = false;
                }
                else
                {
                    var xxx = c[np.array(connected_members[i].ToArray())].flatten();
                    NDarray index = np.where(xxx.not_equals(i))[0];
                    if (np.linalg.matrix_rank(coords[xxx[index]] - coords[i]) < dim)
                    {
                        node_stable[i] = false;
                    }
                }
            }
            return node_stable;
        }

        bool[] Nodal_Stability(NDarray coords, NDarray c, List<int>[] connected_members, bool[] node_stable, int node_i)
        {
            if (connected_members[node_i].Count < dim)
            {
                node_stable[node_i] = false;
            }
            else
            {
                var xxx = c[np.array(connected_members[node_i].ToArray())].flatten();
                NDarray index = np.where(xxx.not_equals(node_i))[0];
                if (np.linalg.matrix_rank(coords[xxx[index]] - coords[node_i]) < dim)
                {
                    node_stable[node_i] = false;
                }
            }
            return node_stable;
        }

        /// <summary>
        /// The Exposure property controls where in the panel a component icon 
        /// will appear. There are seven possible locations (primary to septenary), 
        /// each of which can be combined with the GH_Exposure.obscure flag, which 
        /// ensures the component will only be visible on panel dropdowns.
        /// </summary>
        public override GH_Exposure Exposure => GH_Exposure.primary;

        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// You can add image files to your project resources and access them like this:
        /// return Resources.IconForThisComponent;
        /// </summary>
        protected override System.Drawing.Bitmap Icon => AssemblySequence.Properties.Resources.Assembly_Sequence_icon;

        /// <summary>
        /// Each component must have a unique Guid to identify it. 
        /// It is vital this Guid doesn't change otherwise old ghx files 
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid => new Guid("0C3A93EB-CE9F-4501-8C26-3734D5651A37");
    }
}