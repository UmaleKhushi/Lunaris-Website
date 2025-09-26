// Solution card toggle logic
function toggleSolution(arrow) {
  arrow.classList.toggle("open");
  const solution = document.getElementById("solution-card");
  solution.classList.toggle("show");
}

// Fade-in for normal fade elements on scroll
const fadeElems = document.querySelectorAll('.fade-element');
function fadeInOnScroll() {
  fadeElems.forEach(elem => {
    const rect = elem.getBoundingClientRect();
    if (rect.top < window.innerHeight - 100) {
      elem.classList.add('visible');
    }
  });
}
window.addEventListener('scroll', fadeInOnScroll);
window.addEventListener('load', fadeInOnScroll);

// Fade-in animation for innovation cards (staggered)
window.addEventListener('load', function() {
  const innovationCards = document.querySelectorAll('.innovation-card');
  innovationCards.forEach(function(card, idx){
    setTimeout(function(){
      card.classList.add('visible');
    }, 1000 + idx * 1000); // staggered fade-in
  });
});

// Display uploaded image in Terrain Analysis box (fit perfectly)
const fileInput = document.getElementById('fileUpload');
const analysisArea = document.querySelector('.analysis-area');

fileInput.addEventListener('change', function() {
  const file = this.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      analysisArea.innerHTML = `<img src="${e.target.result}" alt="Terrain Image">`;
    }
    reader.readAsDataURL(file);
  } else {
    analysisArea.innerHTML = '<p>Upload an image to begin analysis</p>';
  }
});
