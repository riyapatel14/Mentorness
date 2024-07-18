document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const age = document.getElementById('age').value;
    const sex = document.getElementById('sex').value;
    const ratings = document.getElementById('ratings').value;
    const leavesUsed = document.getElementById('leaves_used').value;
    const leavesRemaining = document.getElementById('leaves_remaining').value;
    const pastExp = document.getElementById('past_exp').value;
    const tenure = document.getElementById('tenure').value;
    const agePerformance = document.getElementById('age_performance').value;
    const designation = document.getElementById('designation').value;
    
    const data = {
        age: age,
        sex: sex,
        ratings: ratings,
        leaves_used: leavesUsed,
        leaves_remaining: leavesRemaining,
        past_exp: pastExp,
        tenure: tenure,
        age_performance: agePerformance,
        designation: designation
    };
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Predicted Salary: $${data.predicted_salary.toFixed(2)}`;
    })
    .catch(error => console.error('Error:', error));
});
