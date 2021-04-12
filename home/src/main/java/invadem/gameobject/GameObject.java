
package invadem.gameobject;

import processing.core.PApplet;

public abstract class GameObject{
    //initialize the basic attributes
    protected int x;
    protected int y;
    protected int width;
    protected int height;
    protected int velocity;
    protected int health;
    //constructor
    public GameObject(int x, int y, int width, int height,int health,int velocity) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.health = health;
        this.velocity = velocity;
    }
    //get the coordinate
    public int getX(){
        return this.x;
    }
    public int getY(){
        return this.y;
    }
    //check if the the object is alive or not
    public boolean alived(){
        if(this.health <= 0){
            return false;
        }
        return true;
    }
    //get the height,width and health
    public int getHeight() {
        return height;
    }
    public int getWidth() {
        return width;
    }
    public int getVelocity(){
        return velocity;
    }

    public int getHealth() {
        return health;
    }
}
