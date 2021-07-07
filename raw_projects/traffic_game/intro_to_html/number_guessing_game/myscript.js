
let randomNumber = Math.floor( Math.random() * 100 ) +1;       //random number in {1,2,...,100}

const guesses = document.querySelector('.guesses');                 // string containing history of user guesses 
const lastResult = document.querySelector('.lastResult')            // stores the status of the most recent guess
const lowOrHi = document.querySelector('.lowOrHi')

const guessSubmit = document.querySelector('.guessSubmit')
const guessField = document.querySelector('.guessField')

let guessCount = 1         // used to count how many guesses have been made 
let resetButton;           

function checkGuess() {
    let userGuess = Number(guessField.value);   // define variable "userGuess", and fill it with the value in the guessField (also converting the guess value to a number, in case it isn't one) 
    if (guessCount === 1) {
      guesses.textContent = 'Previous guesses: ';    
    }
    guesses.textContent += userGuess + ' ';                             // append latest user input to the guess history string
  
    if (userGuess === randomNumber) {                                  // if current user guess is correct, then do the following
      lastResult.textContent = 'Congratulations! You got it right!';
      lastResult.style.backgroundColor = 'green';
      lowOrHi.textContent = '';
      setGameOver();
    } else if (guessCount === 10) {
      lastResult.textContent = '!!!GAME OVER!!!';
      setGameOver();
    } else {
      lastResult.textContent = 'Wrong!';
      lastResult.style.backgroundColor = 'red';
      if(userGuess < randomNumber) {
        lowOrHi.textContent = 'Last guess was too low!';
      } else if(userGuess > randomNumber) {
        lowOrHi.textContent = 'Last guess was too high!';
      }
    }
  
    guessCount++;             // add 1 to the guess count
    guessField.value = '';    // clear the guess field (current user guess)
    guessField.focus();       // focus on the user input field (i.e. put the cursor there)
  }

guessSubmit.addEventListener('click', checkGuess)     // add an "event listener": when user clicks on guessSubmit button then run function checkGuess()

function setGameOver() {
    guessField.disabled = true;                          // disable the user input field
    guessSubmit.disabled = true;                         // disable the submit button
    resetButton = document.createElement('button');      // create a resetButton html element
    resetButton.textContent = 'Start new game';          // add text to print on the resetButton
    document.body.append(resetButton);                   // add the reset button to the body part of the html
    resetButton.addEventListener('click', resetGame);    // add a listener, which runs the resetGame() function if the user clicks on the resetButton
  }

  function resetGame() {
    guessCount = 1;
  
    const resetParas = document.querySelectorAll('.resultParas p');     // select all <p> elements within the html <div> with class="resultParas" 
    for (let i = 0 ; i < resetParas.length ; i++) {                     // reset all to blank string ''
      resetParas[i].textContent = '';
    }
  
    resetButton.parentNode.removeChild(resetButton);                    // remove the reset buttom 
  
    guessField.disabled = false;                                        // enable the user input field again
    guessSubmit.disabled = false;                                       // enable the user submit button again
    guessField.value = '';                                              // set the user input field to empty string
    guessField.focus();                                                 // "focus" on the user input field
  
    lastResult.style.backgroundColor = 'white';                         // set last result status to white background again 
  
    randomNumber = Math.floor(Math.random() * 100) + 1;
  }

