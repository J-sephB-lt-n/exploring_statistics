
let canvas;
let ctx;
let canvasWidth = 1400;
let canvasHeight = 1000;
let keys = [];                // going to store user key presses as an array, so that multiple key-presses can happen at the same time

document.addEventListener( 'DOMContentLoaded', SetupCanvas );   // when html page has fully loaded, then run the function SetupCanvas()

function SetupCanvas(){
    canvas = document.getElementById('my-canvas');      // get the html element with id "my-canvas" and store it in the js variable "canvas"
    ctx = canvas.getContext('2d');                      // extract the "context" element out of the canvas
    canvas.width = canvasWidth;                         // set canvas width using the globally-defined variable (that we defined at the top of this script)
    canvas.height = canvasHeight;                       // set canvas width using the globally-defined variable (that we defined at the top of this script)
    
    // fill the canvas with black:
    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,canvas.width,canvas.height);       // {0,0} is upper left hand corner co-ords,    {canvas.width,canvas.height} is co-ords of bottom right corner

    // make it so that when a key is pressed down, a "true" value is added to the keys[] array, in the position corresponding to the keycode of the key that was pressed
    document.body.addEventListener( 
                    'keydown', 
                    function(e){
                            keys[e.key] = true;
                            }
    );
    // make it so that when a key is released, that the position in the keys[] array corresponding to that key is set from TRUE to FALSE
    document.body.addEventListener(
                    'keyup',
                    function(e){
                        keys[e.key] = false;
            }
    );
    
    //Render();         // call the render function             

} // end of function SetupCanvas()

// define the ship class:
class Ship {
    // define the attributes of the ship:
    constructor(){
        this.visible = true;            // this attribute determines if ship is visible on screen or not (init value is TRUE)
        // monitor the x and y co-ordinates of the ship:
        // (set initial co-ordinates to the middle of the map)
        this.x = canvasWidth / 2;             // canvas width we defined globally at top of this script
        this.y = canvasHeight / 2;             // canvas height we defined globally at top of this script
        this.movingForward = false;           // indicates if ship is moving forward or not (ship to start stationary on screen)
        this.speed = 0.1;                     // speed
        this.velX = 0;                       // speed in X direction   
        this.velY = 0;                       // speed in Y direction
        this.rotateSpeed = 0.001;
        this.radius = 15;                    // this defines a radius around the ship
        this.angle = 0;                      // angle that ship is pointing
        this.strokeColor = 'white';          // colour used to draw the ship (object is "stroked" rather than "filled")
    }
    // create a function which rotates the ship in a chosen direction:
    Rotate(dir){
        // dir=1 for clockwise
        // dir=-1 for anticlockwise
        this.angle += this.rotateSpeed * dir;
    }
    // create a function which updates the location (co-ordinates) of the ship:
    Update(){
        let radians = this.angle / Math.PI * 180;         // current angle of ship in radians
        // update ship's position using current X velocity, Y velocity and current speed:
        if( this.movingForward ){
            this.velX += Math.cos(radians) * this.speed;
            this.velY += Math.sin(radians) * this.speed;
        }
        // if ship hits the left side of the screen, make it teleport to the far right of the screen:
        if( this.x < this.radius ){ 
            this.x = canvas.width;
        }
        // if ship hits the right of the screen, teleport it to the far left of the screen:
        if( this.x > canvas.width ){ 
            this.x = this.radius;
        }
        // if ship hits the top of the screen ...
        if( this.y < this.radius ){ 
            this.y = canvas.height;
        } 
        // if ship hits the bottom of the screen ...
        if( this.y > canvas.height ){ 
            this.y = this.radius;
        } 

        // reduce the velocity in both directions whenever the update function is called
        // (will cause the ship to slowly decelerate)
        this.velX *= 0.99;        
        this.velY *= 0.99;

        // move the ship according to it's current directional velocities:
        this.x -= this.velX;
        this.y -= this.velY;
    } // end of Update() function
    
    // define a function to draw the ship:
    Draw(){
        ctx.StrokeStyle = this.strokeColor;
        ctx.beginPath();     // start a new drawing of lines
        let vertAngle = ( (Math.PI*2) / 3 );
        let radians = this.angle / Math.PI * 180;
        for( let i=0; i<3; i++ ){
            ctx.lineTo( 
                        this.x - this.radius * Math.cos(vertAngle*i +radians),
                        this.y - this.radius * Math.sin(vertAngle*i +radians) 
                    );
        }
        ctx.closePath();
        ctx.stroke();
    } // end of Draw() function

} // end of class Ship

let ship = new Ship();              // create a new object of class "ship"

function Render(){
    ship.movingForward = keys[87];     // "w" is being pressed, then making movingForward=true
    if( keys[68] ){     // if "d" key is pressed down
        ship.Rotate(1);
    }
    if( keys[65] ){     // if "a" key is pressed down
        ship.Rotate(-1);
    }
    ctx.clearRect(0,0,canvasWidth,canvasHeight);     // clear the canvas
    ship.Update();                                // update ship location
    ship.Draw(); 
    requestAnimationFrame(Render);
}

