
package invadem.gameobject;

import processing.core.PApplet;
import processing.core.PImage;


public class Tank extends GameObject {
    //declare different kinds of images
    private PImage img;
    private PImage heartImg;
    //constructor
    public Tank(PImage img, PImage heartImg,int x, int y, int width, int height,int health,int velocity) {
        super(x, y, width, height,health,velocity);
        this.img = img;
        this.heartImg = heartImg;
    }
    //different tick
    public void leftTick() {
        this.x -= velocity;
    }
    public void rightTick(){
        this.x += velocity;
    }
    //produce the projectile
    public Projectile fire(PImage projectileImage){
        return new Projectile(projectileImage,this.getX()+11,this.getY()+3,1,3,1,1 ,1);
    }
    //tank be attacked
    public void attacked(Projectile p){
        this.health -= p.getDamage();
    }
    //draw the tank
    public void draw(PApplet app) {
        app.image(img, x, y, width, height);
        app.image(heartImg,565,459,25,21);
        if(health > 1){
        app.image(heartImg,590,459,25,21);
    }
        if(health > 2){
        app.image(heartImg,615,459,25,21);
    }
}

}
