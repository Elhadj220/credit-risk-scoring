import { useState } from 'react'
import { predict } from '../api/client'
import ResultCard from './ResultCard'

const DEFAULT_VALUES = {
  LIMIT_BAL: 50000, SEX: 2, EDUCATION: 2, MARRIAGE: 2, AGE: 35,
  PAY_0: 0, PAY_2: 0, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
  BILL_AMT_MEAN: 10000, BILL_AMT_TREND: 0, BILL_AMT_MAX: 20000,
  PAY_AMT_MEAN: 3000, PAY_RATIO: 0.3
}

export default function PredictForm() {
  const [features, setFeatures] = useState(DEFAULT_VALUES)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const handleChange = (key, value) => {
    setFeatures(prev => ({ ...prev, [key]: parseFloat(value) || 0 }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const data = await predict(features)
      setResult(data)
    } catch (err) {
      setError('Erreur lors de la prédiction')
    } finally {
      setLoading(false)
    }
  }

  const fields = [
    { key: 'LIMIT_BAL', label: 'Limite de crédit' },
    { key: 'AGE', label: 'Âge' },
    { key: 'PAY_0', label: 'Retard paiement (mois récent)' },
    { key: 'PAY_2', label: 'Retard paiement (mois -2)' },
    { key: 'BILL_AMT_MEAN', label: 'Solde moyen (6 mois)' },
    { key: 'BILL_AMT_TREND', label: 'Tendance solde' },
    { key: 'PAY_AMT_MEAN', label: 'Paiement moyen' },
    { key: 'PAY_RATIO', label: 'Ratio paiement/solde' },
  ]

  return (
    <div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          {fields.map(({ key, label }) => (
            <div key={key}>
              <label className="text-gray-300 text-sm mb-1 block">{label}</label>
              <input
                type="number"
                step="any"
                value={features[key]}
                onChange={e => handleChange(key, e.target.value)}
                className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          ))}
        </div>

        {error && <p className="text-red-400 text-sm">{error}</p>}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition disabled:opacity-50"
        >
          {loading ? 'Analyse en cours...' : '🔍 Analyser le risque client'}
        </button>
      </form>

      <ResultCard result={result} />
    </div>
  )
}