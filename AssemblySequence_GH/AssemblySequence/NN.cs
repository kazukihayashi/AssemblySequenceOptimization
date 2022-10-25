using Numpy;
using System.Collections.Generic;

namespace GraphEmbedding
{
    struct layer
    {
        public NDarray weight;
        public NDarray bias;
    }
    class GE
    {
        static string directory = Grasshopper.Folders.DefaultAssemblyFolder + @"\AssemblySequence";
        static private int nvi = 0;
        static private int nei = 0;
        static private int neo = 0;
        static private layer[] l = null;
        static private NDarray adjacency = null;
        static private NDarray incidence_1 = null;
        static private NDarray incidence_2 = null;
        static private NDarray incidence_A = null;
        static private NDarray h1;
        static private NDarray h2_0;
        static private NDarray h2;
        static private NDarray h3;
        static private NDarray h4_0;
        static private NDarray nce1;
        static private NDarray nce2;
        static private NDarray h4_1;
        static private NDarray h4_2;
        static private NDarray h4;
        static private NDarray mu;
        static private NDarray mu_sum;
        static private NDarray q;
        static private NDarray y;

        public GE(int n_vertice_input, int n_edge_input, int n_edge_output)
        {
            nvi = n_vertice_input;
            nei = n_edge_input;
            neo = n_edge_output;
        }

        private void connectivity(NDarray c, int nn)
        {
            int nm = c.shape[0];
            adjacency = np.zeros(new int[] { nn, nn });
            for (int i = 0; i < nm; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    adjacency[(int)c[i, j], (int)c[i, (j + 1) % 2]] += 1;
                }
            }
            incidence_1 = np.zeros(new int[] { nn, nm });
            incidence_2 = np.zeros(new int[] { nn, nm });
            for (int i = 0; i < nm; i++)
            {
                incidence_1[(int)c[i, 0], i] += 1;
                incidence_2[(int)c[i, 1], i] += 1;
            }
            incidence_A = incidence_1 + incidence_2;
        }

        private NDarray ActF(NDarray x)
        {
            y = np.copy(x);
            y[y < 0.0] *= 0.2;
            return y;
        }

        private NDarray Forward_layer(layer l_var, NDarray x)
        {
            return np.dot(x, l_var.weight.T)+ l_var.bias;
        }

        private NDarray Mu(NDarray v, NDarray mu, NDarray w, int mu_iter)
        {
            if (mu_iter == 0)
            {
                //h1 = Forward_layer(l[0], w);
                h2_0 = ActF(Forward_layer(l[2], v));
                h2 = Forward_layer(l[1], np.matmul(incidence_A.T, h2_0));
                mu = ActF(h2); //ActF(h1+h2);
            }
            else
            {
                h3 = Forward_layer(l[5], mu);
                h4_0 = np.matmul(incidence_A, mu);
                nce1 = np.clip(np.matmul(adjacency.T, incidence_1).sum(0).repeat(new int[] { neo }).reshape((incidence_A.shape[1], neo)) - 1, (NDarray)1, null);
                nce2 = np.clip(np.matmul(adjacency.T, incidence_2).sum(0).repeat(new int[] { neo }).reshape((incidence_A.shape[1], neo)) - 1, (NDarray)1, null);
                h4_1 = Forward_layer(l[3], np.matmul(incidence_1.T, h4_0) - mu) / nce1;
                h4_2 = Forward_layer(l[3], np.matmul(incidence_2.T, h4_0) - mu) / nce2;
                h4 = Forward_layer(l[4], ActF(h4_1) + ActF(h4_2));
                mu = ActF(h3 + h4);
            }
            return mu;
        }

        private NDarray Q(NDarray mu)
        {
            mu_sum = np.tile(mu.sum(0), new int[] { mu.shape[0], 1 });
            q = Forward_layer(l[6], np.concatenate(new NDarray[] { mu_sum, mu }, 1));
            return q;
        }

        internal NDarray Forward(NDarray v, NDarray w, NDarray c, int n_mu_iter = 3)
        {
            connectivity(c, v.shape[0]);
            mu = np.zeros(w.shape[0], neo);
            for (int i = 0; i < n_mu_iter; i++)
            {
                mu = Mu(v, mu, w, i);
            }
            return Q(mu).flatten();
        }

        public void load()
        {
            l = new layer[7];
            for (int li = 1; li < 7; li++)
            {
                l[li].weight = np.loadtxt(string.Format(@"{0}\l{1}_w.npy", directory, li));
            }
            for (int li = 1; li < 6; li++)
            {
                l[li].bias = np.loadtxt(string.Format(@"{0}\l{1}_b.npy", directory, li));
            }
            l[6].weight = l[6].weight.reshape(new int[] { 1, l[6].weight.size});
            l[6].bias = np.zeros(l[6].weight.shape[0]);
        }
    }

    class Agent
    {
        static private int nvi = 0;
        static private int nei = 0;
        static private int neo = 0;
        static public GE model = null;
        static private NDarray q;
        public Agent(int n_vertice_input, int n_edge_input, int n_edge_output)
        {
            nvi = n_vertice_input;
            nei = n_edge_input;
            neo = n_edge_output;
            model = new GE(nvi, nei, neo);
            model.load();
        }

        public int Decide(NDarray v, NDarray w, NDarray c)
        {
            q = model.Forward(v, w, c);
            return (int)np.argmax(q);
        }
    }
}
