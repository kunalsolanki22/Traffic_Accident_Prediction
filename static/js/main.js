// Form validation
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            const speedLimit = document.getElementById('speed_limit');
            const passengers = document.getElementById('number_of_passengers');
            
            if (speedLimit && speedLimit.value < 0) {
                alert('Speed limit cannot be negative');
                event.preventDefault();
            }
            
            if (passengers && passengers.value < 0) {
                alert('Number of passengers cannot be negative');
                event.preventDefault();
            }
        });
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
}); 