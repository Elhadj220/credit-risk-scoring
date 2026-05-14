export default function ResultCard({ result }) {
  if (!result) return null

  const riskColors = {
    'faible': 'text-green-400 bg-green-900',
    'moyen':  'text-yellow-400 bg-yellow-900',
    'élevé':  'text-red-400 bg-red-900'
  }

  const colorClass = riskColors[result.risk_level] || 'text-gray-400 bg-gray-700'

  return (
    <div className="bg-gray-800 rounded-2xl p-6 mt-6">
      <h2 className="text-white text-xl font-bold mb-4">Résultat de l'analyse</h2>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-700 rounded-xl p-4 text-center">
          <p className="text-gray-400 text-sm mb-1">Score de risque</p>
          <p className="text-4xl font-bold text-white">
            {(result.risk_score * 100).toFixed(1)}%
          </p>
        </div>

        <div className={`rounded-xl p-4 text-center ${colorClass}`}>
          <p className="text-sm mb-1 opacity-70">Niveau de risque</p>
          <p className="text-2xl font-bold uppercase">{result.risk_level}</p>
        </div>

        <div className="bg-gray-700 rounded-xl p-4 text-center col-span-2">
          <p className="text-gray-400 text-sm mb-1">Décision</p>
          <p className={`text-2xl font-bold ${result.prediction === 'défaut' ? 'text-red-400' : 'text-green-400'}`}>
            {result.prediction === 'défaut' ? '⚠️ DÉFAUT PROBABLE' : '✅ CLIENT SAIN'}
          </p>
        </div>
      </div>

      <p className="text-gray-500 text-xs mt-4 text-center">
        Seuil de décision : {result.threshold} | Analysé par : {result.requested_by}
      </p>
    </div>
  )
}