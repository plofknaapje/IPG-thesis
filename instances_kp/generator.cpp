#include <algorithm>
#include <fstream>
#include <random>

using namespace std;

int main(int argc, char **argv) {

  // Type: 0: Potential. 1:Cij positive. 2:Cij both positive and negative.
  for (auto &n : {2, 3}) {
	 for (auto &type : {0, 1, 2}) {
		//25, 50, 100, 150
		for (int m : {75}) {
		  for (auto &ins : {2, 5, 8}) {

			 int                                         R = 100;
			 std::random_device                          rd;
			 static std::mt19937                         engine(rd());
			 std::uniform_int_distribution<unsigned int> dist(1, R);
			 std::uniform_int_distribution<int>          NegativeDist(1, 2 * R);

			 vector<vector<int>> items;

			 std::vector<int> sumWs(n, 0);
			 std::vector<int> b(n, 0);
			 std::vector<int> CPotential(n, 0);

			 for (int i = 0; i < n; ++i)
				CPotential.at(i) = dist(engine);


			 for (int j = 0; j < m; j++) {

				vector<int> item;

				int wij = 0;
				for (int i = 0; i < n; ++i) {
				  // push pij
				  item.push_back(dist(engine));
				  // wij
				  wij = dist(engine);
				  sumWs.at(i) += wij;
				  // push wij
				  item.push_back(wij);
				}

				for (unsigned int p = 0; p < n; ++p) {
				  for (unsigned int o = 0; o < n; ++o) {
					 if (p != o) {
						switch (type) {
						case 0: {
						  item.push_back(CPotential.at(p));
						} break;
						case 1: {
						  // cioj positive
						  item.push_back(dist(engine));
						} break;
						case 2: {
						  // cioj negative or positive
						  item.push_back(NegativeDist(engine) - R);
						} break;
						default: {
						  // cioj negative
						  item.push_back(-dist(engine));
						}
						}
					 }
				  }
				}

				items.push_back(item);
			 }


			 string folder       = "instances_kp/generated/";
			 string instanceName = to_string(n) + "-" + to_string(m) + "-" + to_string(ins) + "-";
			 switch (type) {
			 case 0:
				instanceName += "pot";
				break;
			 case 1:
				instanceName += "cij";
				break;
			 case 2:
				instanceName += "cij-n";
				break;
			 default:
				instanceName += "cij-nn";
			 }
			 instanceName += ".txt";

			 ofstream outFile;
			 outFile.open(folder + instanceName, ios::out);

			 for (int i = 0; i < n; ++i)
				b.at(i) = static_cast<double>(ins) / 10 * sumWs.at(i);

			 outFile << n <<" "<< m << endl;
			 for (int i = 0; i < n; ++i)
				outFile << b.at(i) << (i != (n - 1) ? " " : "\n");

			 for (int j = 0; j < m; j++) {
				outFile << j << " ";
				for (int k = 0; k < items.at(j).size(); k++)
				  outFile << items.at(j).at(k) << (k != (items.at(j).size() - 1) ? " " : "\n");
			 }

			 outFile.close();
		  }
		}
	 }
  }

  return EXIT_SUCCESS;
}
