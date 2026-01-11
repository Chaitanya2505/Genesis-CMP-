// Frontend script for Genesis IPL Predictor
// Replace 'https://your-render-api-url' with your actual Render API URL after deployment

const API_URL = 'https://genesis-cmp.onrender.com'; // Update this after deploying backend on Render

// Function to predict match winner
async function predictMatch(team1, team2, tossWinner, battingFirst, firstInningsScore) {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                team1: team1,
                team2: team2,
                toss_winner: tossWinner,
                batting_first: battingFirst,
                first_innings_score: parseInt(firstInningsScore)
            })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();
        return data.predicted_winner;
    } catch (error) {
        console.error('Error predicting match:', error);
        return 'Error: Unable to predict';
    }
}

// Function to display results in overlay
function displayResults(winner) {
    document.getElementById('winnerText').innerText = winner;
    document.getElementById('overlay').classList.remove('hidden');
}

// Function to close overlay
function closeOverlay() {
    document.getElementById('overlay').classList.add('hidden');
}

// Function to switch views in overlay
function showView(view) {
    // Hide all views
    document.getElementById('viewInsight').style.display = 'none';
    document.getElementById('viewGraphs').style.display = 'none';
    document.getElementById('viewInnings').style.display = 'none';

    // Show selected view
    document.getElementById(`view${view.charAt(0).toUpperCase() + view.slice(1)}`).style.display = 'block';

    // Update tab styles
    document.getElementById('tabInsight').className = 'px-4 py-2 rounded-lg bg-slate-200';
    document.getElementById('tabGraphs').className = 'px-4 py-2 rounded-lg bg-slate-200';
    document.getElementById('tabInnings').className = 'px-4 py-2 rounded-lg bg-slate-200';
    document.getElementById(`tab${view.charAt(0).toUpperCase() + view.slice(1)}`).className = 'px-4 py-2 rounded-lg bg-indigo-600 text-white';
}

// Example usage: Call predictMatch with sample data (replace with actual form inputs)
document.addEventListener('DOMContentLoaded', () => {
    // Sample prediction on page load (for testing)
    predictMatch('MI', 'CSK', 'MI', 'MI', 150).then(winner => {
        displayResults(winner);
    });
});