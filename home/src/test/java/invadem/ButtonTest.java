package invadem;

import invadem.gameobject.*;
import org.junit.Test;


import static org.junit.Assert.*;

public class ButtonTest {

    @Test
    public void testButtonConstruction() {
        Button button = new Button(null,null,240,300,150,39,1,0);
        assertNotNull(button);
    }
    @Test
    public void testButtonGet() {
        Button button = new Button(null,null,240,300,150,39,1,0);
        assertEquals(240, button.getX());
        assertEquals(300, button.getY());
        assertEquals(150,button.getWidth());
        assertEquals(39,button.getHeight());
        assertEquals(1,button.getHealth());
        assertEquals(0,button.getVelocity());
    }
}