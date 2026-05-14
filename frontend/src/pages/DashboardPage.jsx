import PredictForm from '../components/PredictForm'

export default function DashboardPage({ onLogout }) {
  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-3xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Credit Risk Scoring</h1>
            <p className="text-gray-400">Modèle Random Forest — ROC-AUC: 0.77</p>
          </div>
          <button
            onClick={onLogout}
            className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition"
          >
            Déconnexion
          </button>
        </div>

        <div className="bg-gray-800 rounded-2xl p-6">
          <h2 className="text-white text-xl font-semibold mb-6">
            Analyser un client
          </h2>
          <PredictForm />
        </div>
      </div>
    </div>
  )
}