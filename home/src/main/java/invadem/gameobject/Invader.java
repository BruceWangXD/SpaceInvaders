package invadem.gameobject;

import processing.core.PApplet;
import processing.core.PImage;

public class Invader extends GameObject{
    //declare different kinds of images
    protected PImage imgOne;
    protected PImage imgTwo;
    protected PImage projectileImage;
    //declare the score for the invaders
    protected int score;
    //constructor
    public Invader(PImage imgOne, PImage imgTwo, PImage projectileImage, int x, int y, int width, int height,int health,int velocity,int score) {
        super(x, y, width, height,health, velocity);
        this.imgOne = imgOne;
        this.imgTwo = imgTwo;
        this.projectileImage = projectileImage;
        this.score = score;
    }
    //different tick
    public void leftTick() {
        this.x -= velocity;
    }
    public void rightTick(){
        this.x += velocity;
    }
    public void downTick(){
        this.y += velocity;
    }
    //draw the invader
    public void draw(PApplet app, boolean down) {
        if(!down){
            app.image(imgOne, x, y, width, height);
        }else if(down){
            app.image(imgTwo, x, y, width, height);
        }

    }
    //produce the projectile
    public Projectile fire(){
        Projectile p = new Projectile(projectileImage,x+8,y-16,1,3,3,-1,1);
        return p;
    }
    //when the invader is attacked
    public void attacked(){
        this.health--;
    }
    //get the score of the invaders
    public int getScore() {
        return score;
    }
}
