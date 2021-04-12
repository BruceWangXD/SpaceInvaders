package invadem;

import invadem.gameobject.*;
import processing.core.PApplet;
import processing.core.PFont;
import processing.core.PImage;


import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import java.io.File;
import java.util.ArrayList;


public class App extends PApplet {
    PImage a;
    //Declare all the Image
    PImage tankImage;
    PImage projectileImage;
    PImage powerProjectileImage;
    PImage gameOverImage;
    PImage nextLevelImage;
    PImage normalInvader1;
    PImage normalInvader2;
    PImage powerInvader1;
    PImage powerInvader2;
    PImage armouredInvader1;
    PImage armouredInvader2;
    PImage heartImage;
    //Declare the font
    PFont regular;
    //Declare the tank
    Tank myTank;
    //Declare the Button
    Button startButton;
    //Declare the ArrayList for the invaders
    ArrayList<Invader> invaders = new ArrayList<>();
    //Declare the ArrayList for the projectiles
    ArrayList<Projectile> projectiles = new ArrayList<>();
    //Declare the ArrayList for the barriers
    ArrayList<Barrier> barriers = new ArrayList<>();
    //Declare all the clip for the sound
    Clip gameOverClip;
    Clip markUpClip;
    Clip fireClip;
    Clip winClip;
    Clip attackedClip;
    Clip backClip;
    //Initialize the button is not pressed
    boolean pressed = false;
    //Initialize the tank does not go left or right
    boolean goLeft = false;
    boolean goRight = false;
    //Initialize the invader does not go down at first
    boolean down = false;
    //Initialize the game does not over at first
    boolean gameOver = false;
    //Initialize the game does go to next level page at first
    boolean nextLevel = false;
    //Initialize the game does start at first
    boolean gameStart = false;
    //Initialize the default highest score
    int highScore = 10000;
    //Initialize the current score as 0
    int currentScore = 0;
    //Initialize the step as 0
    int step = 0;
    //Initialize the time as 0
    int time = 0;
    //Initialize the restartTime as 0
    int restartTime = 0;
    //Initialize the level as 1
    int level = 1;
    //initialize the frequency of invaders firing as 5 second which is 300
    int gap = 300;

    public App() {
        //Set up your object
    }

    public void setup() {
        //initialize the frameRates as 60
        frameRate(60);
        //initialize the Image for the tank and its heart
        tankImage = loadImage("tank1.png");
        heartImage = loadImage("heart.png");
        //initialize the tank
        restartTank();
        //initialize the start button
        startButton = new Button(loadImage("button1.png"),loadImage("button2.png"),240,300,150,39,1,0);
        //initialize the Image for all kinds of invaders
        normalInvader1 = loadImage("invader1.png");
        normalInvader2 = loadImage("invader2.png");
        powerInvader1 = loadImage("invader1_power.png");
        powerInvader2 = loadImage("invader2_power.png");
        armouredInvader1 = loadImage("invader1_armoured.png");
        armouredInvader2 = loadImage("invader2_armoured.png");
        //initialize the Image for the empty barrier
        PImage empty = loadImage("empty.png");
        //initialize all kinds of barrier components
        BarrierComponent solid = new BarrierComponent(loadImage("barrier_solid1.png"),loadImage("barrier_solid2.png"),loadImage("barrier_solid3.png"),empty,0,0,8,8,3,0);
        BarrierComponent top = new BarrierComponent(loadImage("barrier_top1.png"),loadImage("barrier_top2.png"),loadImage("barrier_top3.png"),empty,0,0,8,8,3,0);
        BarrierComponent right = new BarrierComponent(loadImage("barrier_right1.png"),loadImage("barrier_right2.png"),loadImage("barrier_right3.png"),empty,0,0,8,8,3,0);
        BarrierComponent left = new BarrierComponent(loadImage("barrier_left1.png"),loadImage("barrier_left2.png"),loadImage("barrier_left3.png"),empty,0,0,8,8,3,0);
        //initialize all the barrier
        barriers.add(new Barrier(200,430,left, right, solid, top));
        barriers.add(new Barrier(308,430,left, right, solid, top));
        barriers.add(new Barrier(416,430,left, right, solid, top));
        //initialize the Image for the projectile
        projectileImage = loadImage("projectile.png");
        powerProjectileImage = loadImage("projectile_lg.png");
        //initialize the Image for the game over page and next level page
        gameOverImage = loadImage("gameover.png");
        nextLevelImage = loadImage("nextlevel.png");
        //initialize the invaders
        restartInvaders();
        //initialize the all the music and sound
        gameOverClip = setSound("gameOver.wav");
        markUpClip = setSound("markUp.wav");
        fireClip = setSound("fire.wav");
        winClip = setSound("win.wav");
        attackedClip = setSound("attacked.wav");
        backClip = setSound("background.wav");
        //start the background music as the game begin
        backClip.loop(100);
        backClip.start();
        //initialize the font
        regular = createFont("PressStart2P-Regular.ttf",20);
    }

    public void settings() {
        size(640, 480);
    }

    public void draw() {
        //Main Game Loop

        background(0);
        //if the start button is pressed and released the game should start
        if (gameStart){
            //set up the size for the font
            textFont(regular,10);
            //the game is normally running
            if(!gameOver && !nextLevel){
                //draw the boundary line
                stroke(255);
                line(180, 0, 180, 480);
                stroke(255);
                line(460, 0, 460, 480);
                //draw the font
                text(String.format("Current score \n%d",currentScore), 10, 30);
                text(String.format("High score \n%d",highScore), 530, 30);
                text(String.format("Level %d",level), 10, 470);
                //check the collision of all projectile
                Collision(projectiles);
                //if it's time for random invaders to fire
                if(time % gap == 0 && time > 0){
                    int randomInvader = (int)(Math.random() * invaders.size());
                    projectiles.add(invaders.get(randomInvader).fire());
                }
                //draw all the invaders
                for(int i = 0; i < invaders.size(); i++){
                    if(invaders.get(i).getY()+16 > 430){
                        gameOver = true;
                    }
                    invaders.get(i).draw(this,down);
                    if(time % 2 == 0){
                        if(step % 76 < 30){
                            invaders.get(i).rightTick();
                            down = false;
                        }else if(step % 76 < 38){
                            invaders.get(i).downTick();
                            down = true;
                        }else if(step % 76 < 68){
                            invaders.get(i).leftTick();
                            down = false;
                        }else if(step % 76 < 76){
                            invaders.get(i).downTick();
                            down = true;
                        }
                    }
                }
                //increase the step
                if(time % 2 == 0){
                    step++;
                }
                //when the left key is pressed and tank isn't reach the left boundary
                if(goLeft == true && goRight != true && myTank.getX() > 180){
                    myTank.leftTick();
                    //when the right key is pressed and tank isn't reach the right boundary
                }else if(goLeft != true && goRight == true && myTank.getX() < 438){
                    myTank.rightTick();
                }
                //draw the tank
                myTank.draw(this);
                for(Barrier b:barriers){
                    b.draw(this);
                }
                //check the projectile
                projectilesCheck(projectiles);
                //draw the projectile
                projectilesDraw(projectiles);
                //if all the invaders is dead, go to next level page
                if(invaders.size() <= 0){
                    nextLevel = true;
                }
                //if the tank is dead, go to game over page
                if(!myTank.alived()){
                    gameOver = true;
                }
                //increase time
                time++;
                //if the game goes to next level page
            }else if(nextLevel){
                image(nextLevelImage,264,232,112,16);
                if(restartTime == 0){
                    winClip.setFramePosition(5);
                    winClip.loop(0);
                    winClip.start();
                    backClip.stop();
                }
                restartTime++;
                if(restartTime == 120){
                    winClip.stop();
                    level++;
                    gap -= 60;
                    if(gap < 60){
                        gap = 60;
                    }
                    restartEveryThing();
                    nextLevel = false;
                }
            }else if(gameOver){
                image(gameOverImage, 264, 232, 112, 16);
                if(restartTime == 0){
                    gameOverClip.setFramePosition(5);
                    gameOverClip.loop(0);
                    gameOverClip.start();
                    backClip.stop();
                }
                restartTime++;
                if(restartTime == 120){
                    gameOverClip.stop();
                    level = 1;
                    gap = 300;
                    if(currentScore > highScore){
                        highScore = currentScore;
                    }
                    restartEveryThing();
                    currentScore = 0;
                    gameOver = false;
                }
            }
            //if the start button isn't pressed and released the game should stay at start page
        }else{
            textFont(regular,50);
            text(String.format("TANK WAR"), 105, 150);
            startButton.draw(this,pressed);
        }


    }
    //start the app
    public static void main(String[] args) {
        PApplet.main("invadem.App");
    }
    //check all the collision for the projectile
    public void Collision(ArrayList<Projectile> projectiles){
        boolean ac = false;
        for(int i = 0; i < projectiles.size(); i++){
            ac = false;
            if(projectiles.get(i).isFriendly()){
                for(int invaderNumber = 0; invaderNumber < invaders.size(); invaderNumber++){
                    if(projectiles.get(i).intersect(invaders.get(invaderNumber))) {
                        invaders.get(invaderNumber).attacked();
                        if(!invaders.get(invaderNumber).alived()) {
                            currentScore += invaders.get(invaderNumber).getScore();
                            markUpClip.setFramePosition(5);
                            markUpClip.loop(0);
                            markUpClip.start();
                            invaders.remove(invaderNumber);
                            invaderNumber--;
                        }
                        ac = true;
                    }
                }
            }else{
                if(projectiles.get(i).intersect(myTank)){
                    myTank.attacked(projectiles.get(i));
                    ac = true;
                }
            }
            for(int barrierNumber = 0; barrierNumber < barriers.size(); barrierNumber++){
                for(int BCNumber = 0; BCNumber < barriers.get(barrierNumber).getCs().size(); BCNumber++){
                    if(projectiles.get(i).intersect(barriers.get(barrierNumber).getCs().get(BCNumber)) && barriers.get(barrierNumber).getCs().get(BCNumber).alived()){
                        barriers.get(barrierNumber).getCs().get(BCNumber).attacked(projectiles.get(i));
                        attackedClip.setFramePosition(10);
                        attackedClip.loop(0);
                        attackedClip.start();
                        ac = true;
                    }
                }
            }
            if(ac){
                projectiles.remove(i);
                i--;
            }
        }
    }

    //check if the projectile is going out of the screen
    public void projectilesCheck(ArrayList<Projectile> projectiles){
        if(projectiles.size() > 0) {
            if(projectiles.get(0).getY() < 0 || projectiles.get(0).getY() > 480){
                projectiles.remove(0);
            }
        }
    }
    //function to draw all the projectile
    public void projectilesDraw(ArrayList<Projectile> projectiles){
        for(int i = 0; i < projectiles.size(); i++){
            projectiles.get(i).draw(this);
        }
    }
    //function to restart everything
    public void restartEveryThing(){
        backClip.loop(100);
        backClip.start();
        restartTime = 0;
        time = 0;
        step = 0;
        restartTank();
        restartInvaders();
        restartBarriers();
        restartProjectile();
    }
    //function to restart invaders
    public void restartInvaders(){
        invaders.clear();
        for(int column = 0; column < 10; column++){
            invaders.add(new Invader(armouredInvader1,armouredInvader2,projectileImage,180+column*26,0,16,16,3,1,250));
        }
        for(int column = 0; column < 10; column++){
            invaders.add(new PowerInvader(powerInvader1,powerInvader1,powerProjectileImage,180+column*26,24,16,16,1,1,250));
        }
        for(int row = 0; row < 2; row++){
            for(int column = 0; column < 10; column++){
                invaders.add(new Invader(normalInvader1,normalInvader2,projectileImage,180+column*26,48+row*24,16,16,1,1,100));
            }
        }
    }
    //function to restart tank
    public void restartTank(){
        myTank = new Tank(tankImage, heartImage,309, 464, 22, 16, 3,1);
    }
    //function to restart barriers
    public void restartBarriers(){
        for(int barrierNumber = 0; barrierNumber < barriers.size(); barrierNumber++){
            for(int BCNumber = 0; BCNumber < barriers.get(barrierNumber).getCs().size(); BCNumber++){
                barriers.get(barrierNumber).getCs().get(BCNumber).setHealth(3);
            }
        }
    }
    //function to restart projectile
    public void restartProjectile(){
        projectiles.clear();
    }
    //deal with the situation when the key pressed
    public void keyPressed(){
        if(key == CODED) {
            //if the right key is pressed make the tank go right
            if (keyCode == RIGHT) {
                goRight = true;
                //if the left key is pressed make the tank go left
            } else if (keyCode == LEFT ) {
                goLeft = true;
            }
        }
    }
    //deal with the situation when the key released
    public void keyReleased(){
        //if the space key is released make the tank fire
        if(key == ' '){
            projectiles.add(myTank.fire(projectileImage));
            fireClip.setFramePosition(10);
            fireClip.loop(0);
            fireClip.start();
        }else if(key == CODED) {
            //if the right key is released stop the tank moving right
            if (keyCode == RIGHT) {
                goRight = false;
                //if the left key is released stop the tank moving left
            } else if (keyCode == LEFT) {
                goLeft = false;
            }
        }
    }
    //check if the mouse was press at the proper position
    public void mousePressed() {
        if(mouseX > 240 && mouseX < 390 && mouseY > 300 && mouseY < 319){
            pressed = true;
        }
    }
    //the mouse is pressed and release the game start
    public void mouseReleased(){
        if(pressed){
            gameStart = true;
        }
    }
    //the function to set up the sound
    public Clip setSound(String name) {
        try {
            File a = new File("src\\main\\resources\\"+name);
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(a);
            Clip clip = AudioSystem.getClip();
            clip.open(audioInputStream);
            return clip;
        } catch(Exception ex) {
            System.out.println("Error with playing sound.");
            ex.printStackTrace();
            return null;
        }
    }
}
